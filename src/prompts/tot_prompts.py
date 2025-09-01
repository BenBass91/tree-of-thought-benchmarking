"""
Tree of Thought prompt templates for structured reasoning
"""

TOT_MATH_TEMPLATE = """
Problem: {problem}

Using Tree of Thought methodology, solve this step-by-step:

1. **Problem Analysis**: What type of problem is this and what do we need to find?

2. **Solution Approaches**: Consider 3 different ways to solve this:
   - Approach A: [Brief description]
   - Approach B: [Brief description] 
   - Approach C: [Brief description]

3. **Approach Evaluation**: Which approach seems most promising and why?

4. **Step-by-Step Solution**: Execute your chosen approach with clear steps.

5. **Verification**: Check your answer and explain why it's correct.

Provide your complete reasoning and final answer.
"""

TOT_LOGIC_TEMPLATE = """
Logic Problem: {problem}

Apply Tree of Thought reasoning:

1. **Problem Understanding**: Identify the logical structure and constraints.

2. **Reasoning Paths**: Explore multiple logical approaches:
   - Path 1: [Method description]
   - Path 2: [Method description]
   - Path 3: [Method description]

3. **Path Selection**: Choose the most reliable reasoning path.

4. **Logical Deduction**: Work through the logic step by step.

5. **Consistency Check**: Verify your reasoning is sound and consistent.

Show all your reasoning work and provide the final answer.
"""

TOT_GENERAL_TEMPLATE = """
Question: {problem}

Using systematic Tree of Thought analysis:

1. **Problem Breakdown**: What are the key components of this question?

2. **Multiple Perspectives**: Consider different angles:
   - Perspective A: [Description]
   - Perspective B: [Description]
   - Perspective C: [Description]

3. **Best Approach**: Select the most effective perspective and explain why.

4. **Detailed Analysis**: Work through your chosen approach thoroughly.

5. **Final Synthesis**: Combine insights and provide a comprehensive answer.

Present your complete thought process and conclusion.
"""

TOT_CODING_TEMPLATE = """
Coding Problem: {problem}

Apply Tree of Thought methodology to solve this programming challenge:

1. **Problem Understanding**: What exactly needs to be accomplished?

2. **Algorithm Approaches**: Consider multiple solution strategies:
   - Approach A: [Algorithm/data structure]
   - Approach B: [Alternative method]
   - Approach C: [Different perspective]

3. **Complexity Analysis**: Evaluate time and space complexity for each approach.

4. **Optimal Solution**: Choose the best approach and explain your reasoning.

5. **Implementation Strategy**: Break down the implementation steps.

6. **Edge Cases**: Consider potential edge cases and error handling.

Provide your complete reasoning and solution approach.
"""
