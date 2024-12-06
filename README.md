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

## Details of Adequacy Evaluation Agent
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


                                                                                                                                                                                                                                                                                                                                                                           
