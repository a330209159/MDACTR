# MDACTR
This is a repository for Data Availability of Our Paper:
Multi-dimensional Assessment of CrowdSourced Testing Reports via LLMs

# Framework
![Framework of Our Apporach](https://raw.githubusercontent.com/a330209159/MDACTR/main/framework.png)
# Implementation Details of Proprocessing
## Details of Defect Classification Agent
This agent implements an automated defect clustering and statistical analysis process based on a large model, which is divided into three stages: defect clustering, defect statistics, and defect statistics sorting. Below are the details of the algorithm's implementation:

### 1. Defect Clustering
First, the program extracts all the defect descriptions from a specified Excel file and passes these defect descriptions along with the functional points list to the large model, asking the model to cluster the defects according to the functional points. The clustering details are as follows:
- Each defect is categorized based on its relevance to the functional points.
- The clustering result includes two fields: "Defect Name" and "Defect Brief Description". The "Defect Name" represents the category name after clustering (e.g., "xx Issue" or "xx Functionality Anomaly"), while the "Defect Brief Description" summarizes the main issues in that category.

Specific operations:
- Read defect descriptions from an Excel file, remove line breaks, and construct a complete defect list.
- Read the functional points list from a text file, which serves as the basis for clustering.
- The large model clusters the defects based on the functional points list.

### 2. Defect Statistics
After clustering is complete, the program counts the number of defects in each category. By matching the clustering result with the original defect descriptions, a statistical table containing "Defect Name", "Defect Description", and "Defect Count" is generated.

Specific operations:
- Generate a defect count for each defect category based on the clustering result and original defect descriptions.
- Output the result as a table with headers: "Defect Name", "Defect Description", and "Defect Count".

### 3. Defect Statistics Sorting
Next, the program sorts the statistical table based on defect count, rearranging the rows in descending order of defect count.

Specific operations:
- Sort the defect statistics table by the "Defect Count" column in descending order.
- Output the sorted defect statistics table.

### Execution Flow
1. **Get Defect List**: Extract defect descriptions from the Excel file.
2. **Get Functional Points List**: Read functional points from the text file.
3. **Cluster Defects**: Use the large model to cluster defects based on the functional points.
4. **Count Defects**: Calculate the defect count for each category based on the clustering result.
5. **Sort Defect Statistics**: Sort the defect statistics table by defect count.

### Prompts
| Prompt Name              | Prompt Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **cluster_execute**       | Please cluster the defects in the software testing reports according to the granularity of the functional points list. The defects will be provided as a list, where each line represents a defect description:<br>**<defects_list>**<br>The functional points list is as follows:<br>**<functional_points_list>**<br>Please cluster the defects based on the functional points list, and output a table with two main fields: 'Defect Name' and 'Defect Brief Description'. 'Defect Name' should represent the clustered category and be described in the format of 'xx problem' or 'xx functionality abnormality', while 'Defect Description' should briefly summarize the types of issues included in this clustered category. Do not output the full defect descriptions that are included in this cluster. Ensure that each defect is accurately clustered into the corresponding functional point and defect type, and keep the clustering granularity as detailed as possible for each functional point. Avoid having a 'miscellaneous' category, and clustering can have single defects; only include the issues reflected in the defect list, do not invent any new defects. Only output the table with defect categories and their descriptions, and do not output any unrelated content. |
| **defect_statistics**     | Based on the following table, count the number of defects for each defect category, and output the results as a table with headers 'Defect Name', 'Defect Description', and 'Defect Quantity'. Ensure that all defects in the defect list are counted, and the quantity is correct.<br>**<clustering_result>**<br>The defect list is as follows:<br>**<defects_list>**<br>Only output the defect count table, and do not output any unrelated content, such as explanations.                                                                                                                                                                                                 |
| **defect_amount_reorder** | Sort the rows of the following defect statistics table by the 'Defect Quantity' column, in descending order, and output the table again.<br>**<statistics_result>**<br>Only output the defect count table, and do not output any unrelated content, such as explanations.                                                                                                                                                                                                                                                                                                                                                                                                                       
# Implementation Details of Adequacy Dimension Assessment

## Details of Adequacy Assessment Agent
This agent implements an automated process for evaluating the adequacy dimension of software testing reports, based on a large language model (LLM). The evaluation is divided into several steps: functional point extraction, test case processing, coverage calculation, and final adequacy scoring. Below are the details of the algorithm's implementation:

### 1. Functional Point Extraction
First, the program loads a list of functional points from a specified text file. The functional points represent the different components or features of the software that need to be tested.

Specific operations:
- The functional points list is read from a text file, with each functional point listed on a new line.

### 2. Test Case Processing
The program then extracts the test cases from an Excel file, which includes the test case name and its description. The test cases are then processed to match them against the functional points.

Specific operations:
- Read test case descriptions from an Excel file, ensuring the correct columns (i.e., "testcase name" and "testcase description") are present.
- Construct a list of formatted test cases for further processing.

### 3. Coverage Calculation
The program then calculates which functional points are covered by the test cases. For each test case, the program uses the LLM to classify it according to the relevant functional points.

Specific operations:
- For each test case, the LLM is prompted to match the test case with the corresponding functional points.
- The output is a list of functional points that the test case covers.

### 4. Adequacy Scoring
Finally, the program calculates the adequacy score based on the coverage of functional points. The adequacy score is computed as the ratio of covered functional points to total functional points, multiplied by 100. The program then generates a comment based on the coverage rate.

Specific operations:
- Calculate the coverage rate (i.e., the ratio of covered functional points to total functional points).
- Generate a comment that summarizes the coverage situation.
- Output the adequacy score and the comment as a JSON object.

### Execution Flow
1. **Set Functional Points Path**: Define the path to the functional points file.
2. **Read Test Case Data**: Extract test case names and descriptions from the Excel file.
3. **Match Test Cases to Functional Points**: Use the LLM to classify test cases by functional points.
4. **Calculate Coverage**: Determine the covered functional points for each test case.
5. **Calculate Adequacy Score**: Compute the adequacy score and generate a feedback comment.
6. **Store Results**: Save the results (score and comment) to a CSV file for further analysis.

### Prompts

| Prompt Name               | Prompt Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **coverage_calculate**     | There is a test requirement document with the following list of functional points, totaling {} items:<br>**<functional_points_list>**<br>There is also a list of test cases written by testers, totaling {} items:<br>**<test_case_list>**<br>Please process each test case in the following steps:<br>1. Classify the defects in the test case according to the functional points list, output the functional point that corresponds to the test case, and the name of the functional point must exactly match the name in the list. Do not create new names for functional points. Process each test case without omission.<br>2. Collect all functional points covered by each test case and output as a list, counted as covered functional points. Construct a JSON output with the covered functional points list, key as 'covered'. Wrap the JSON output using ```json``` format. |
| **coverage_rate_score**    | In a software test, the complete list of functional points is as follows, totaling {} items:<br>**<functional_points_list>**<br>And the list of functional points covered by the test cases written by testers is as follows:<br>**<covered_functional_points_list>**<br>Please process these two functional point lists:<br>1. Calculate the total number of functional points in the complete list.<br>2. Calculate the number of functional points covered by the test cases.<br>3. Calculate the coverage rate by dividing the number of covered points by the total number of points.<br>4. Based on the coverage rate, provide three sentences of feedback to inform the developers about the coverage of the test cases, including which points are covered and which are not.<br>Please output the JSON with the coverage rate score (coverage rate * 100) under 'score', and the feedback under 'comment'. Wrap the JSON output using ```json``` format. |
| **adequacy_statistics**    | For the following list of functional points coverage by test cases, please generate a table with 'Test Case Name', 'Adequacy Score', and 'Feedback'. Ensure that the adequacy score is calculated based on the coverage rate and that feedback is concise and informative, reflecting the adequacy of the coverage.<br>**<test_case_coverage_list>**<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# Implementation Details of Textual Dimension Assessment

## Details of Textual Assessment Agent
This agent implements an automated process for evaluating the textual dimension of software testing reports, based on a set of predefined scoring rules. The evaluation is divided into several steps: test case extraction, score calculation, and final textual scoring. Below are the details of the algorithm's implementation:

### 1. Test Case Extraction
First, the program loads test case data from an Excel file, which includes test case descriptions and other relevant fields (such as test case name, priority, expected results, etc.).

**Specific operations:**
- The program extracts test case data from an Excel file, ensuring the correct columns (i.e., "testcase id", "testcase name'", "priority", "testcase description", etc.) are present.
- The data is converted into a list of dictionaries for further processing.

### 2. Textual Scoring Rule Loading
The program reads the textual scoring rules from a text file. These rules define how each aspect of the test case text should be evaluated (e.g., format, completeness, clarity).

**Specific operations:**
- Read the textual scoring rules from a file (the content of this file has been shown in 'Rule of Textual Dimension Assessment').
- Parse the rules to apply them to the test case descriptions.

### 3. Score Calculation
The program processes each test case by applying the textual scoring rules. For each test case, the program assigns scores for different criteria (e.g., RM1, RM2, RR1, etc.) based on the rules.

**Specific operations:**
- For each test case, the program uses predefined rules to evaluate different textual aspects of the test case.
- The program calculates scores for each rule and aggregates them into a total score.

### 4. Total Score Calculation
The program calculates the total score for a test case by summing the individual scores for each rule (e.g., RM1, RM2, etc.). It then generates a detailed feedback comment based on the individual scores.

**Specific operations:**
- Calculate the total score by summing the individual scores for each rule.
- Generate a feedback comment explaining the score for each rule and overall performance.

### 5. Average Score Calculation
The program can calculate the average score for a set of test cases. This helps in understanding the overall quality of the test cases in terms of textual completeness and correctness.

**Specific operations:**
- Calculate the average score for each rule across multiple test cases.
- Output the average score for each rule, providing insights into the overall textual quality of the test cases.

### Execution Flow
1. **Set Test Case Data Path**: Define the path to the test case Excel file.
2. **Read Test Case Data**: Extract test case data from the Excel file.
3. **Load Scoring Rules**: Load the textual scoring rules from the file.
4. **Calculate Individual Scores**: Apply the scoring rules to each test case.
5. **Calculate Total Score**: Aggregate the individual scores into a total score for each test case.
6. **Calculate Average Score**: Compute the average score across all test cases.
7. **Store Results**: Save the results (scores and comments) to a CSV file for further analysis.

### Prompts

| Prompt Name               | Prompt Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **textual_scoring_rules**  | Below are the scoring criteria for writing test cases. This standard requires each aspect of a test case to be evaluated for compliance. The compliance scoring criteria cover RM, RR, and RA, with each category specifying the scoring rules for each point. <br>**<scoring_criteria_details>**<br>Please use the above scoring criteria to evaluate the compliance of this test case. The score results should be output in JSON format, where the `criteria` field represents the point number, `score` is the score, and `reason` is the justification for the given score. An example of the JSON format is as follows: <br>[<br>{"criteria":"RM1",<br>  "reason":"<explanation_of_reason>",<br>  "score":"3"},<br>{"criteria":"RM2",<br>  "reason":"<explanation_of_reason>",<br>  "score":"4"},<br>…<br>] |
| **textual_testcase_score** | Below is a software test case. Please score it according to the above criteria and output the result in JSON format. Do not output any unrelated content. <br>**<test_case_details>** |

## Rule of Textual Dimension Assessment
| **Criteria**            | **Rule**                                                                                           | **Scoring**                                                                                                                                             |
|-------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **RM1 (Size)**           | The length of the test case description should be between 10 and 100 words.                         | - 3 points: Fully meets the length requirement. <br> - Points are deducted based on the proportion of words over or under the specified range.           |
| **RM2 (Readability)**    | The description should be concise, smooth, and easy to understand.                                  | - 2 points: Fully meets the readability requirement. <br> - Points are deducted for logical errors or unclear sentence structure.                        |
| **RM3 (Punctuation)**    | Correct usage of punctuation marks according to grammar standards.                                 | - 3 points: Fully meets the punctuation requirement. <br> - Deduct 0.25 points for each punctuation error, down to 0 points.                             |
| **RR1 (Step-by-Step Clarity)**  | The test steps should be numbered logically, each step clear and concise.                        | - 4 points: Fully meets the requirement. <br> - Deduct 1 point for missing steps or disorganized sequence. <br> - 2 points if steps are not numbered.     |
| **RR1.1 (Step Numbering or Bullet Usage)** | Use consistent numbering or bullet points (e.g., numbered list or '-' bullets).          | - 1 point: Fully meets the requirement. <br> - Deduct 0.25 points for each inconsistency or error in bullet/numbering usage.                              |
| **RR2 (Environment)**    | Complete information about the test environment (e.g., hardware specs, OS version, emulator, etc.). | - 3 points: Fully meets the requirement. <br> - Deduct 0.5 points for each missing or incomplete detail.                                                 |
| **RR3 (Preconditions)**  | All preconditions that need to be met before running the test should be listed clearly.             | - 2 points: Fully meets the requirement. <br> - Deduct 0.5 points for missing or unclear preconditions. <br> - 0 points if key preconditions are missing. |
| **RR4 (Expected Results)** | The expected results should be clearly specified.                                                | - 2 points: Fully meets the requirement. <br> - Deduct 1-2 points for unclear or missing expected results.                                              |
| **RR5 (Additional Information)** | The additional information should be complete, including test case designer, priority, etc.   | - 2 points: Fully meets the requirement. <br> - Deduct points proportionally for each missing key piece of information.                                  |
| **RA1 (Interface Elements)** | The description of interface elements (e.g., buttons, links, input fields) should be accurate.   | - 5 points: Fully meets the requirement. <br> - Deduct 1 point for each unclear or missing description of an element.                                      |
| **RA2 (User Actions)**   | The interaction process between the user and the interface elements should be described in detail.  | - 5 points: Fully meets the requirement. <br> - Deduct 1 point for each missing or inaccurate description of user actions.                               |

## Implementation Details of Competitive Dimension Assessment

### Details of Competitive Assessment Agent
This agent implements an automated process for evaluating the competitive dimension of software testing reports using a large language model (LLM). The evaluation process includes defect classification, scoring, and generating feedback based on predefined defect categories. Below are the details of the algorithm's implementation:

### 1. Defect Classification and Scoring Table Creation
First, the agent loads the defect category statistics and scoring rules, and generates a scoring table for defects. This involves categorizing defects based on their descriptions, according to predefined rules. The resulting scoring table is then used to evaluate the defects in testing reports.

Specific operations:
- Read the defect category statistics from a text file.
- Load the defect scoring rules from a separate file.
- Generate a defect scoring table using the category statistics and rules.

### 2. Defect Report Evaluation
The agent processes defect reports from Excel files. Each defect description is matched against the scoring table, categorized, and assigned a score. Finally, the scores are summed up and feedback is provided on the coverage of defect categories.

Specific operations:
- Read the defect descriptions from an Excel file.
- Match each defect to its corresponding category from the scoring table.
- Calculate the total score based on the categories and their respective scores.
- Provide feedback on which defect categories are covered and which are not, including suggestions for improvements.

### 3. Score Calculation and Feedback Generation
The score for each report is calculated by summing the individual defect category scores. The feedback provides insights into the coverage of defect categories and gives recommendations for test case improvements.

Specific operations:
- Sum up the scores of categorized defects.
- Generate feedback based on the coverage of defect categories.
- Output the score and feedback in JSON format.

### 4. Reporting and Saving Results
Finally, the agent processes multiple defect reports from a folder, evaluates them, and saves the results in a CSV file. The CSV includes the report filename, score, and feedback.

Specific operations:
- Read defect reports from a folder.
- Evaluate each report using the scoring table and feedback mechanism.
- Save the results (filename, score, and feedback) in a CSV file for further analysis.

### Execution Flow
1. **Set Defect Cluster Table Path**: Define the path to the defect category statistics file.
2. **Generate Defect Scoring Table**: Create a scoring table using defect category statistics and scoring rules.
3. **Evaluate Individual Report**: For each report, extract defects, categorize them, and calculate scores.
4. **Generate Feedback**: Provide feedback on defect category coverage and improvement suggestions.
5. **Store Results**: Save the results in a CSV file for further analysis.

### Prompts

| Prompt Name               | Prompt Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **scoring_table_create**   | Please generate a defect scoring table based on the following defect category statistics table and scoring rules.<br>**<defect_category_statistics>**<br>Scoring table generation rules:<br>**<scoring_rules>**<br>Provide the complete calculation process for each step used in the generation of the scoring table, and finally output the result in table form. |
| **table_extract**          | Please extract only the table generated from Step 4, without any additional unrelated content or explanations.<br>**<generated_scoring_table>** |
| **score_report**           | Below is a defect scoring table that defines defect categories and their respective scores:<br>**<defect_scoring_table>**<br>Now, here is a list of defect descriptions in the report:<br>**<defect_descriptions>**<br>Follow these steps to process the defects in the report:<br>1. Classify each defect according to the scoring table.<br>2. Find the score for each defect category and output the score. Unclassified defects do not receive any score.<br>3. Sum the scores for all defects and output the total score.<br>4. Provide feedback on the coverage of defect categories and suggest improvements.<br>Output the results in JSON format with `score` and `comment` as the keys. |
| **competitive_statistics** | For the following list of defects evaluated in the report, please generate a table with 'Report Name', 'Score', and 'Feedback'. The score should be based on the defect category coverage and the feedback should be concise and informative.<br>**<defect_report_coverage_list>** |

### Steps for Creating a Scoring Table

#### Step 1: Define Preliminary Scores
Based on the defect statistics table, calculate preliminary scores for each defect category. The base score is set at 100 points. The formula for calculating preliminary scores is:

$$\text{Preliminary Score} = \frac{\text{Maximum Base Score}}{\text{Defect Count} + 1}$$

For example, if a defect is reported 5 times and the maximum base score is set at 100 points, then the preliminary score is:

$$\text{Preliminary Score} = \frac{100}{5 + 1} \approx 16.67$$


#### Step 2: Calculate Total Score
Add up the preliminary scores of all defect categories to get the total score. The formula for calculating the total score is:

$$\text{Total Score} = \sum \text{Preliminary Scores of All Defects}$$

#### Step 3: Calculate Normalized Scores
For each defect category, divide the preliminary score by the total score and then multiply by 100 to get the normalized score. You need to iterate through each defect category to calculate and output the calculation process and result for the normalized score. The formula for calculating normalized scores is:

$$\text{Normalized Score} = \left( \frac{\text{Preliminary Score}}{\text{Total Score}} \right) \times 100$$

For example, if a defect's preliminary score is 16.67 and the total score is 346, then the normalized score is:

$$\text{Normalized Score} = \left( \frac{16.67}{346} \right) \times 100 \approx 4.82$$

#### Step 4: Generate the Scoring Table
Finally, create a table listing all defect categories and fill in the normalized scores for each category. The table header should have two fields: "Defect Name", "Defect Description", and "Normalized Score". The table should be clear, easy to read, and understandable. The table is created only in Step 4; other steps do not require the output of a table.

| Defect Name | Defect Description | Normalized Score |
|-------------|--------------------|------------------|
| ...         | ...                | ...              |



                                                                                                                                                                                                                                                                                                                                                                           
