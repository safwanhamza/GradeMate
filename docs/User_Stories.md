#   GradeMate - User Stories

This document outlines the user stories for the GradeMate application, a system designed to streamline and automate the grading of exams.

##   User Stories

<details>
<summary>Login to the system</summary>

* **Priority:** Highest
* **User Story:** As an examiner, I want to log in to the system, so that I can access the grading features.
* **Acceptance Criteria:**
    * Given that the examiner provides valid credentials, when they log in, then the system should authenticate and redirect them to the homepage.
    * Given invalid credentials, when the examiner attempts to log in, then an error message should be displayed.
</details>

<details>
<summary>Upload Exam Files</summary>

* **Priority:** Highest
* **User Story:** As an examiner, I want to upload exam documents in PDF, image, or DOC formats, so that the system can process them for grading.
* **Acceptance Criteria:**
    * Given that the document is in a supported format, when the examiner uploads it, then the file should be processed and displayed for review.
    * Given an unsupported format, when the examiner tries to upload it, then the system should show an error message.
</details>

<details>
<summary>Parse and extract answers from uploaded exam files</summary>

* **Priority:** Highest
* **User Story:** As an examiner, I want the system to automatically extract student answers and model keys from uploaded exam documents i.e. PDFs or images.
* **Acceptance Criteria:**
    * Given that an examiner uploads an exam document, when the system processes the document, then it should:
        * Accurately identify and extract relevant answer sections from the document.
        * Convert text from images into digital text.
        * Clean and preprocess the extracted text to improve accuracy.
        * Store the extracted answers in a suitable format for further analysis.
</details>

<details>
<summary>Define Custom Grading Guidelines</summary>

* **Priority:** Highest
* **User Story:** As an examiner, I want to define grading guidelines such as length, structure, and semantic similarity, so that the system evaluates answers according to my preferences.
* **Acceptance Criteria:**
    * Given that grading criteria are set, when the examiner saves the guidelines, then the system should store them and display a confirmation.
    * Given incomplete guidelines, when the examiner tries to save them, then the system should prompt to complete all required fields.
</details>

<details>
<summary>Grading of Multiple-Choice Questions (MCQs)</summary>

* **Priority:** Highest
* **User Story:** As an Educator, I want the system to grade multiple-choice questions (MCQs) in exams. The system should evaluate MCQs by comparing student responses to a predefined answer key. Feedback should highlight correct and incorrect answers for students., So that I can reduce time spent on manual grading and provide faster feedback to students.
* **Acceptance Criteria:**
    * Given that MCQs are included in the exam and key is provided, when the system processes the answers, then it should:
        * Grade all MCQs with 100% accuracy.
        * Deliver results with clear feedback on correct and incorrect answers.
</details>

<details>
<summary>Grading of Open-Ended Questions (OEQs)</summary>

* **Priority:** Highest
* **User Story:** As an educator, I want the system to grade students' subjective or open-ended responses. The system should: Assess the quality of the response based on factors like relevance, clarity, and depth of understanding, Length and structure of the response. Compare the student's response to model answers and grading guidelines., Provide detailed feedback to students on their performance., So that I can ensure consistent and objective grading without bias while saving time spent on manual grading.
* **Acceptance Criteria:**
    * Given that examiner has provided model answers and guidelines, when the system processes students’ scripts, then it should:
        * Evaluate based on predefined key points, length, semantics, and similarity.
        * Provide personalized feedback with at least 95% grading accuracy.
</details>

<details>
<summary>Feedback on Graded Answers</summary>

* **Priority:** Medium
* **User Story:** As an examiner, I want the system to generate feedback for each graded answer, so that students can understand their mistakes
* **Acceptance Criteria:**
    * Given a graded answer, when the system generates feedback, then it should display meaningful comments for improvement.
    * Given an ungraded answer, when feedback is attempted, then the system should notify the examiner to grade the answer first.
</details>

<details>
<summary>Export Grading Report</summary>

* **Priority:** Medium
* **User Story:** As an examiner, I want to download a detailed report of the results, so that I can share it with students or other stakeholders.
* **Acceptance Criteria:**
    * Given graded answers, when the examiner requests a report, then the system should generate and download it in a structured format.
    * Given no graded answers, when the examiner requests a report, then the system should prompt to complete grading first.
</details>
