// static/js/exam_upload.js
document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const uploadForm = document.getElementById('exam-upload-form');
    const statusDiv = document.getElementById('upload-status');
    const extractedQuestionsDiv = document.getElementById('extracted-questions');
    const questionsContainer = document.getElementById('questions-container');
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('exam-file');
        if (!fileInput.files.length) {
            alert('Please select a file to upload.');
            return;
        }
        
        const file = fileInput.files[0];
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            alert('Only PDF files are supported.');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading indicator
        statusDiv.style.display = 'block';
        extractedQuestionsDiv.style.display = 'none';
        
        // Upload the file and process
        fetch('/upload-exam/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            statusDiv.style.display = 'none';
            
            if (data.status === 'error') {
                alert(`Error: ${data.message}`);
                return;
            }
            
            // Display extracted questions and answers
            if (data.questions_and_answers && data.questions_and_answers.length > 0) {
                displayExtractedQuestions(data.questions_and_answers);
            } else {
                alert('No questions or answers were extracted from the PDF.');
            }
        })
        .catch(error => {
            statusDiv.style.display = 'none';
            alert(`Error: ${error.message}`);
        });
    });
    
    // Function to display extracted questions and answers
    function displayExtractedQuestions(questionsAndAnswers) {
        questionsContainer.innerHTML = '';
        
        questionsAndAnswers.forEach(item => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'question-item';
            
            questionDiv.innerHTML = `
                <h3>Question ${item.question_no}</h3>
                <div class="question-statement">
                    <strong>Question:</strong>
                    <p>${item.question_statement}</p>
                </div>
                <div class="student-answer">
                    <strong>Student's Answer:</strong>
                    <p>${item.complete_answer}</p>
                </div>
            `;
            
            questionsContainer.appendChild(questionDiv);
        });
        
        extractedQuestionsDiv.style.display = 'block';
    }
});