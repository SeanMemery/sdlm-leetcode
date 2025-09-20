import json
from typing import List, Dict, Any
from datasets import load_dataset

def load_leetcode_dataset(max_items: int = 20, split: str = "test", difficulty: str = "Easy") -> List[Dict[str, Any]]:
    """
    Load LeetCodeDataset from Hugging Face.
    
    Args:
        max_items: Maximum number of problems to load
        split: Dataset split to use ("train" or "test")
        difficulty: Filter by difficulty level ("Easy", "Medium", "Hard")
        
    Returns:
        List of problem dictionaries with keys:
        - task_id: Problem identifier
        - problem_description: Problem statement
        - starter_code: Initial code template
        - test: Test function for evaluation
        - input_output: Test cases
    """
    dataset = load_dataset('newfacade/LeetCodeDataset')
    problems = dataset[split]
    
    # Filter by difficulty
    if difficulty:
        problems = problems.filter(lambda x: x['difficulty'] == difficulty)
    
    # Limit to max_items
    if max_items and max_items < len(problems):
        problems = problems.select(range(max_items))
    
    return list(problems)

def build_evaluator():
    """
    Build local evaluator that runs test cases directly using the test field from LeetCodeDataset.
    Returns a callable that takes (code, problem) and returns (accepted, status).
    """
    class LocalEvaluator:
        def evaluate(self, code: str, problem: Dict[str, Any]) -> tuple[bool, str]:
            """
            Evaluate code against problem test cases using input_output field.
            
            Args:
                code: Python code to evaluate (should contain the solution function)
                problem: Problem dictionary with test cases from LeetCodeDataset
                
            Returns:
                (accepted, status_message)
            """
            try:
                # Create execution environment with necessary imports
                exec_globals = {
                    '__builtins__': __builtins__,
                    'List': list,  # For type hints
                    'Optional': type(None),  # For Optional type hints
                }
                
                # Execute the code (should define the solution function)
                exec(code, exec_globals)
                
                # Get the entry point function (e.g., twoSum)
                entry_point = problem['entry_point']
                if entry_point not in exec_globals:
                    return False, f"Function '{entry_point}' not found in solution"
                
                solution_func = exec_globals[entry_point]
                
                # Test against input_output cases
                input_output_cases = problem['input_output']
                for i, test_case in enumerate(input_output_cases):
                    input_str = test_case['input']
                    expected_output = test_case['output']
                    
                    try:
                        # Parse the input string to extract function arguments
                        # Example: "nums = [3,3], target = 6" -> solution_func([3,3], 6)
                        input_dict = {}
                        exec(input_str, {}, input_dict)
                        
                        # Call the solution function with the parsed arguments
                        if entry_point == 'twoSum':
                            result = solution_func(input_dict['nums'], input_dict['target'])
                        else:
                            # Generic handling for other functions - extract all variables
                            args = list(input_dict.values())
                            result = solution_func(*args)
                        
                        # Convert result to string for comparison
                        result_str = str(result) if result is not None else "None"
                        
                        # Compare with expected output
                        if result_str != expected_output:
                            return False, f"Wrong Answer on test case {i+1}: expected {expected_output}, got {result_str}"
                            
                    except Exception as e:
                        return False, f"Runtime Error on test case {i+1}: {str(e)}"
                
                return True, "Accepted"
                    
            except Exception as e:
                return False, f"Runtime Error: {str(e)}"
    
    return LocalEvaluator()