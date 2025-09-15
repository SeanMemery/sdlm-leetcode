# sdlm/leetcode/evaluators/leetcode_eval.py

import hashlib
import os
from typing import Tuple
from diskcache import Cache

# Import leetcode_env components inside methods to avoid circular imports

class LeetCodeEvaluator:
    def __init__(self, cache_dir="./cache_leetcode"):
        """Initialize LeetCode evaluator using leetcode_env.
        
        Requires LEETCODE_SESSION and LEETCODE_CSRF_TOKEN environment variables.
        """
        # Verify environment variables are set
        if "LEETCODE_SESSION" not in os.environ:
            raise ValueError("LEETCODE_SESSION environment variable must be set")
        if "LEETCODE_CSRF_TOKEN" not in os.environ:
            raise ValueError("LEETCODE_CSRF_TOKEN environment variable must be set")
        
        try:
            # Monkey patch gym with gymnasium for NumPy 2.0 compatibility
            import sys
            import gymnasium
            sys.modules['gym'] = gymnasium
            
            from leetcode_env.environment import LeetCodeEnv
            self.env = LeetCodeEnv()
        except Exception as e:
            raise Exception("Install leetcode-hard-gym: pip install git+https://github.com/GammaTauAI/leetcode-hard-gym.git") from e
            
        self.cache = Cache(cache_dir)

    def _format_code(self, code: str) -> str:
        """Format code by removing unnecessary elements."""
        # Simple cleanup - remove print statements and assertions
        import re
        code = re.sub(r'\bprint\(.*?\)', '', code)
        code = re.sub(r'\bassert\b.*', '', code)
        return code.strip()


    def submit_for_evaluation(self, code: str, slug: str):
        """Submit code to LeetCode for evaluation using leetcode_env."""
        try:
            from leetcode_env.utils.formatting import PythonSubmissionFormatter
            from leetcode_env.types import LeetCodeSubmission, ProgrammingLanguage
        except Exception as e:
            raise Exception("Install leetcode-hard-gym: pip install git+https://github.com/GammaTauAI/leetcode-hard-gym.git") from e

        # Validate code before formatting - but don't fail immediately
        try:
            # Basic syntax validation
            import ast
            ast.parse(code)
        except SyntaxError as e:
            print(f"Info: Syntax error in code for {slug}, will submit anyway for LeetCode feedback: {e}")
            # Continue to LeetCode submission - let LeetCode provide feedback

        try:
            # Format code for LeetCode submission
            code = PythonSubmissionFormatter.to_leetcode(code).strip()
            code = self._format_code(code)
        except Exception as e:
            print(f"Info: Code formatting failed for {slug}: {e}, submitting raw code")
            # Continue with raw code - let LeetCode handle it

        # Create submission object
        submission = LeetCodeSubmission(
            code=code,
            lang=ProgrammingLanguage.PYTHON3,
            question_slug=slug,
            timeout=10
        )

        try:
            # Use leetcode_env to submit and get results
            observation, reward, done, info = self.env.step(submission)
            
            # Parse results from leetcode_env response
            if observation and hasattr(observation, 'last_submission_result'):
                result = observation.last_submission_result
                return {
                    "status_msg": "Accepted" if result.status == "Accepted" else result.status,
                    "total_correct": getattr(result, 'total_correct', 0),
                    "total_testcases": getattr(result, 'total_testcases', 0),
                    "status_runtime": getattr(result, 'runtime_ms', -1)
                }
            else:
                # Fallback if observation structure is different
                return {
                    "status_msg": "Accepted" if reward > 0 else "Wrong Answer",
                    "total_correct": 1 if reward > 0 else 0,
                    "total_testcases": 1,
                    "status_runtime": -1
                }
        except Exception as e:
            print(f"Submission error for {slug}: {e}")
            return {
                "status_msg": "Runtime Error",
                "total_correct": 0,
                "total_testcases": 0,
                "status_runtime": -1
            }

    def check_if_in_cache_or_submit(self, task_id: str, code: str):
        """Check cache first, then submit if not found."""
        key = hashlib.sha256(f"{task_id}__{code}".encode()).hexdigest()
        fields = [f"{key}_{s}" for s in ("run_success","total_correct","total_tests","runtime")]

        if fields[0] in self.cache:
            success = self.cache[fields[0]]
            total_correct = self.cache[fields[1]]
            total_tests = self.cache[fields[2]]
            runtime = self.cache[fields[3]]
            print(f"Cached: {task_id} Success={success} {total_correct}/{total_tests}")
            return success, total_correct, total_tests, runtime

        res = self.submit_for_evaluation(code, task_id)
        success = res.get("status_msg") == "Accepted"
        total_correct = res.get("total_correct", 0)
        total_tests = res.get("total_testcases", 0)
        runtime = res.get("status_runtime", -1)

        self.cache[fields[0]] = success
        self.cache[fields[1]] = total_correct
        self.cache[fields[2]] = total_tests
        self.cache[fields[3]] = runtime

        print(f"LC Eval: {task_id} Success={success} {total_correct}/{total_tests}")
        return success, total_correct, total_tests, runtime