"""
ProPS-DMP Agent.

Uses an LLM (Prompted Policy Search) to search over DMP weight vectors.
Same goal as CMA-ES, SAC-DMP, and PPO-DMP: find the weight vector that
produces the best crawling gait.

Core idea (from ProPS paper, Zhou et al. NeurIPS 2025):
    At each iteration the LLM receives the full history of
    (weights, cost/reward) pairs as in-context examples. It proposes
    a new weight vector together with a textual justification
    ("textual gradient"). The ProPS+ variant adds a semantic
    description of the environment and task.

Two modes:
    - ProPS  (numeric only): template has weights + cost history.
    - ProPS+ (semantic):     template adds Go2 crawl environment
                             description, DMP/PCA explanation, etc.

Two objective views (both tracked, selectable via config):
    - cost minimization:  cost = -reward  (LLM minimizes)
    - reward maximization: reward directly (LLM maximizes)

Two feedback detail levels (selectable via config):
    - basic:    weights + cost/reward only
    - detailed: weights + cost/reward + distance_x + terminated +
                PCA latent range

Architecture:
    - Jinja2 templates format the prompt with history + task info.
    - LLMBrain handles the LLM API calls (OpenAI, Gemini, Anthropic).
    - WeightHistoryBuffer stores (weights, cost, reward, metadata) pairs
      and formats them as text for the prompt.
    - The runner drives the loop: asks for weights, does the MuJoCo
      rollout with Go2CrawlEnv + DMPPolicy, feeds results back.

Runner interface:
    - generate_random_weights()        : for warmup phase
    - propose_weights()                : LLM proposes new weights
    - store_result(weights, cost, ...) : add to history buffer
    - update_best(weights, reward)     : track best found
    - get_best()                       : return best weights/reward
    - save() / load()                  : checkpointing
"""

import os
import re
import time
import numpy as np
from jinja2 import Environment, FileSystemLoader

from agent.propsbuffernobias import WeightHistoryBuffer


# ---------------------------------------------------------------------------
# LLM Brain — handles API calls to OpenAI, Gemini, Anthropic, Ollama
# ---------------------------------------------------------------------------

class LLMBrain:
    """
    Thin wrapper around LLM API calls.

    Supports:
        - OpenAI    (gpt-4o, o1-preview, o3-mini, etc.)
        - Gemini    (gemini-2.5-pro, gemini-2.5-flash, etc.)
        - Anthropic (claude-3-7-sonnet, etc.)
        - Ollama    (llama3:70b, gpt-oss:120b, mistral, etc.)
                    Uses OpenAI-compatible API at a custom base_url.
    """

    def __init__(self, model_name, max_retries=5, retry_delay=60,
                 provider=None, ollama_base_url=None):
        """
        Parameters
        ----------
        model_name : str
            Model identifier. Examples:
                "gpt-4o", "claude-3-7-sonnet-20250219",
                "gemini-2.5-pro-preview-05-06", "gpt-oss:120b"
        max_retries : int
            Max retries per API call on failure.
        retry_delay : int
            Base seconds between retries.
        provider : str or None
            Explicit provider override. One of:
                "openai", "gemini", "anthropic", "ollama"
            If None, auto-detected from model_name.
        ollama_base_url : str or None
            Base URL for the Ollama instance.
            e.g. "https://sol-hpc.example.edu:11434"
            Only used when provider is "ollama".
            Can also be set via OLLAMA_BASE_URL env var.
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Determine provider — explicit override or auto-detect
        if provider is not None:
            self.provider = provider
        elif "gemini" in model_name:
            self.provider = "gemini"
        elif "claude" in model_name:
            self.provider = "anthropic"
        else:
            # Default to openai for gpt-* models.
            # Ollama must be set explicitly via provider config.
            self.provider = "openai"

        # Ollama base URL
        self.ollama_base_url = ollama_base_url

        # Lazy-init clients
        self._client = None

    def _get_client(self):
        """Initialize the API client on first use."""
        if self._client is not None:
            return self._client

        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self._client = genai

        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Client(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )

        elif self.provider == "ollama":
            from openai import OpenAI
            base_url = self.ollama_base_url or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434",
            )
            # Ollama exposes OpenAI-compatible API at /v1
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            self._client = OpenAI(
                base_url=base_url,
                api_key="ollama",  # Ollama doesn't need a real key
            )

        else:
            # OpenAI cloud
            from openai import OpenAI
            self._client = OpenAI()

        return self._client

    def query(self, prompt):
        """
        Send prompt to the LLM. Returns response text.

        Retries on failure with exponential backoff.

        Parameters
        ----------
        prompt : str
            The full prompt text.

        Returns
        -------
        response_text : str
        api_time : float
            Wall-clock seconds for the API call.
        """
        client = self._get_client()

        for attempt in range(self.max_retries):
            try:
                api_start = time.time()

                if self.provider == "gemini":
                    model = client.GenerativeModel(
                        model_name=self.model_name,
                    )
                    response = model.generate_content(prompt)
                    text = response.text

                elif self.provider == "anthropic":
                    message = client.messages.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=2048,
                    )
                    text = message.content[0].text

                else:
                    # OpenAI and Ollama use the same client interface
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    text = completion.choices[0].message.content

                api_time = time.time() - api_start
                return text, api_time

            except Exception as e:
                print(f"LLM query attempt {attempt + 1}/{self.max_retries} "
                      f"failed: {e}")
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (attempt + 1)
                    print(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM query failed after {self.max_retries} "
                        f"attempts: {e}"
                    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_weights(response_text, num_weights):
    """
    Parse the LLM response to extract the weight vector.

    Tries multiple formats:
        1. w[0]: 1.234; w[1]: -0.567; ... (ProPS format)
        2. [1.234, -0.567, ...] (array format)
        3. Any line with enough float numbers

    Also extracts reasoning (everything after the weights line).

    Parameters
    ----------
    response_text : str
        Raw LLM response.
    num_weights : int
        Expected number of weights.

    Returns
    -------
    weights : np.ndarray, shape (num_weights,)
    reasoning : str
        The LLM's explanation (lines after the weights).

    Raises
    ------
    ValueError if parsing fails.
    """
    lines = response_text.strip().split("\n")

    # --- Strategy 1: w[i]: val format ---
    for line_idx, line in enumerate(lines):
        pattern = re.compile(
            r"w\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        )
        matches = pattern.findall(line)
        if len(matches) >= num_weights:
            weights = np.zeros(num_weights)
            for idx_str, val_str in matches[:num_weights]:
                idx = int(idx_str)
                if idx < num_weights:
                    weights[idx] = float(val_str)
            reasoning = "\n".join(lines[line_idx + 1:]).strip()
            return weights, reasoning

    # --- Strategy 2: JSON-style array [val, val, ...] ---
    for line_idx, line in enumerate(lines):
        bracket_match = re.search(r"\[([^\]]+)\]", line)
        if bracket_match:
            inner = bracket_match.group(1)
            numbers = re.findall(
                r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", inner,
            )
            if len(numbers) >= num_weights:
                weights = np.array(
                    [float(x) for x in numbers[:num_weights]]
                )
                reasoning = "\n".join(lines[line_idx + 1:]).strip()
                return weights, reasoning

    # --- Strategy 3: any line with enough numbers ---
    for line_idx, line in enumerate(lines):
        numbers = re.findall(
            r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line,
        )
        if len(numbers) >= num_weights:
            weights = np.array(
                [float(x) for x in numbers[:num_weights]]
            )
            reasoning = "\n".join(lines[line_idx + 1:]).strip()
            return weights, reasoning

    raise ValueError(
        f"Could not parse {num_weights} weights from LLM response:\n"
        f"{response_text[:1000]}"
    )


# ---------------------------------------------------------------------------
# ProPS-DMP Agent
# ---------------------------------------------------------------------------

class ProPSDMPAgent:

    def __init__(
        self,
        num_weights,
        initial_weights,
        total_iterations,
        n_bfs,
        n_components,
        action_range=2.0,
        llm_model_name="gpt-4o",
        llm_provider=None,
        ollama_base_url=None,
        template_dir="templates",
        template_name="props_numeric.j2",
        env_description_file=None,
        feedback_mode="basic",
        objective="cost",
        buffer_size=200,
        warmup_episodes=10,
        optimum=None,
        step_size=0.1,
        max_retries=5,
        retry_delay=60,
    ):
        """
        Parameters
        ----------
        num_weights : int
            Dimensionality of DMP weight vector (n_components * n_bfs).
        initial_weights : np.ndarray
            Initial DMP weights from the circle trajectory fit.
            Search bounds are centered on these values.
        total_iterations : int
            Total LLM optimization iterations (not counting warmup).
        n_bfs : int
            Number of DMP basis functions per dimension.
        n_components : int
            Number of PCA dimensions (DMP dimensions).
        action_range : float
            Search range: initial_weights ± this value.
        llm_model_name : str
            LLM model identifier. Examples:
                "gpt-4o", "gpt-4o-mini", "o3-mini-2025-01-31",
                "gemini-2.5-pro-preview-05-06",
                "gemini-2.5-flash-preview-04-17",
                "claude-3-7-sonnet-20250219",
                "gpt-oss:120b" (Ollama), "llama3:70b" (Ollama)
        llm_provider : str or None
            Explicit provider override. One of:
                "openai", "gemini", "anthropic", "ollama"
            If None, auto-detected from model_name.
            Must be set to "ollama" for local/Sol Ollama models.
        ollama_base_url : str or None
            Base URL for the Ollama server.
            e.g. "https://sol-hpc.example.edu:11434"
            Can also be set via OLLAMA_BASE_URL env var.
        template_dir : str
            Directory containing Jinja2 prompt templates.
        template_name : str
            Filename of the prompt template to use.
            "props_numeric.j2" for ProPS, "props_semantic.j2" for ProPS+.
        env_description_file : str or None
            Filename of the environment description template (for ProPS+).
            Relative to template_dir. e.g. "go2_crawl_description.j2".
            If None, the semantic block is omitted.
        feedback_mode : str
            "basic" — weights + f(w) only.
            "detailed" — weights + f(w) + dist_x + terminated + PCA range.
        objective : str
            "cost" — LLM minimizes f(w) = -reward.
            "reward" — LLM maximizes f(w) = reward.
        buffer_size : int
            Max entries in the history buffer. Oldest dropped when full.
        warmup_episodes : int
            Number of random rollouts before the first LLM call.
        optimum : float or None
            Expected optimal value. Tells the LLM when to explore vs exploit.
            If None, not included in the prompt.
        step_size : float
            Suggested exploration step size for the LLM.
        max_retries : int
            Max retries per LLM API call on failure.
        retry_delay : int
            Base seconds between retries (multiplied by attempt number).
        """
        self.num_weights = num_weights
        self.n_bfs = n_bfs
        self.n_components = n_components
        self.initial_weights = np.asarray(
            initial_weights, dtype=np.float64,
        ).flatten()
        self.total_iterations = total_iterations
        self.action_range = action_range
        self.feedback_mode = feedback_mode
        self.objective = objective
        self.warmup_episodes = warmup_episodes
        self.optimum = optimum
        self.step_size = step_size

        # Weight bounds
        self.w_low = self.initial_weights - action_range
        self.w_high = self.initial_weights + action_range

        # LLM
        self.llm_brain = LLMBrain(
            model_name=llm_model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            provider=llm_provider,
            ollama_base_url=ollama_base_url,
        )

        # Jinja2 template
        # Resolve template_dir relative to project root if it's a relative path
        if not os.path.isabs(template_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_dir = os.path.join(project_root, template_dir)
        jinja_env = Environment(loader=FileSystemLoader(template_dir))
        # get_template expects a bare filename, not an absolute path
        self.template = jinja_env.get_template(os.path.basename(template_name))

        # Load env description if provided (for ProPS+)
        self.env_description = None
        if env_description_file:
            env_desc_template = jinja_env.get_template(
                os.path.basename(env_description_file),
            )
            self.env_description = env_desc_template.render()

        # History buffer
        self.buffer = WeightHistoryBuffer(max_size=buffer_size)

        # Tracking
        self.best_reward = -np.inf
        self.best_weights = None
        self.best_cost = np.inf
        self.iterations_done = 0
        self.total_api_time = 0.0

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def generate_random_weights(self):
        """
        Generate a random weight vector within bounds.
        Used during warmup phase to seed the history buffer.

        Returns
        -------
        weights : np.ndarray, shape (num_weights,)
        """
        return np.random.uniform(low=self.w_low, high=self.w_high)

    # ------------------------------------------------------------------
    # LLM proposal
    # ------------------------------------------------------------------

    def propose_weights(self):
        """
        Ask the LLM to propose a new DMP weight vector.

        Builds the prompt from the template + history buffer,
        queries the LLM, parses the response.

        Returns
        -------
        weights : np.ndarray, shape (num_weights,)
            Proposed weight vector (clipped to bounds).
        reasoning : str
            The LLM's textual justification.
        full_prompt : str
            The complete prompt sent to the LLM.
        raw_response : str
            The raw LLM response text.
        api_time : float
            Wall-clock seconds for the API call.
        """
        # Format history based on feedback mode
        if self.feedback_mode == "detailed":
            feedback_text = self.buffer.format_detailed(
                self.num_weights, objective=self.objective,
            )
        else:
            feedback_text = self.buffer.format_basic(
                self.num_weights, objective=self.objective,
            )

        # Build template variables
        template_vars = {
            "num_weights": self.num_weights,
            "N_BFS": self.n_bfs,
            "n_components": self.n_components,
            "MAX_ITERS": self.total_iterations,
            "iter_idx": self.iterations_done,
            "feedback_text": feedback_text,
            "step_size": self.step_size,
            "objective": self.objective,
            "w_low": self.w_low,
            "w_high": self.w_high,
            "initial_weights": self.initial_weights,
        }

        # Optional fields
        if self.optimum is not None:
            template_vars["optimum"] = self.optimum

        if self.env_description is not None:
            template_vars["env_description"] = self.env_description

        # Render prompt
        prompt = self.template.render(template_vars)

        # Query LLM
        raw_response, api_time = self.llm_brain.query(prompt)
        self.total_api_time += api_time

        # Parse response
        weights, reasoning = parse_weights(raw_response, self.num_weights)

        # Clip to bounds
        weights = np.clip(weights, self.w_low, self.w_high)

        return weights, reasoning, prompt, raw_response, api_time

    # ------------------------------------------------------------------
    # Result storage
    # ------------------------------------------------------------------

    def store_result(self, weights, reward, metadata=None):
        """
        Store a rollout result in the history buffer.

        Parameters
        ----------
        weights : np.ndarray
            The DMP weight vector that was evaluated.
        reward : float
            Total reward from the rollout.
        metadata : dict or None
            Extra info: distance_x, terminated, steps,
            pca_x_range, pca_y_range.
        """
        cost = -reward
        self.buffer.add(weights, cost, reward, metadata)

    # ------------------------------------------------------------------
    # Best tracking
    # ------------------------------------------------------------------

    def update_best(self, weights, reward):
        """Track the best weights found so far (by reward)."""
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_weights = np.array(weights, dtype=np.float64).copy()
            self.best_cost = -reward

    def get_best(self):
        """Return (best_weights, best_reward)."""
        return (
            self.best_weights.copy() if self.best_weights is not None else None,
            self.best_reward,
        )

    def get_best_cost(self):
        """Return (best_weights, best_cost) — lowest cost found."""
        return (
            self.best_weights.copy() if self.best_weights is not None else None,
            self.best_cost,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_state(self):
        """Return state dict for checkpointing compatibility."""
        return {
            "best_weights": self.best_weights,
            "best_reward": float(self.best_reward),
            "best_cost": float(self.best_cost),
            "iterations_done": self.iterations_done,
            "total_api_time": self.total_api_time,
        }

    def save(self, path):
        """
        Save agent state to disk.

        Creates:
            {path}_meta.npz     — best weights, reward, cost, iteration count
            {path}_buffer.npz   — full history buffer
        """
        np.savez(
            path + "_meta.npz",
            best_weights=(
                self.best_weights
                if self.best_weights is not None
                else np.array([])
            ),
            best_reward=float(self.best_reward),
            best_cost=float(self.best_cost),
            iterations_done=self.iterations_done,
            total_api_time=self.total_api_time,
        )

        self.buffer.save(path + "_buffer.npz")

    def load(self, path):
        """
        Load agent state from disk.

        Expects:
            {path}_meta.npz     — metadata
            {path}_buffer.npz   — history buffer
        """
        meta_path = path + "_meta.npz"
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            bw = meta["best_weights"]
            self.best_weights = bw if bw.size > 0 else None
            self.best_reward = float(meta["best_reward"])
            self.best_cost = float(meta["best_cost"])
            self.iterations_done = int(meta["iterations_done"])
            self.total_api_time = float(meta["total_api_time"])

        buffer_path = path + "_buffer.npz"
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)