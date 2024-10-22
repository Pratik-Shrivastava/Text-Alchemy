const container = document.getElementById("container");
const registerbtn = document.getElementById("register");
const loginbtn = document.getElementById("login");

registerbtn.addEventListener("click", () => {
  container.classList.add("active");
});

loginbtn.addEventListener("click", () => {
  container.classList.remove("active");
});

// Get references to the input and button elements
const submitButton = document.getElementById('submitButton');
const userEmail = document.getElementById('emailId');
const responseMessage = document.getElementById('responseMessage');
const otpContainer = document.getElementById('otpcontainer');
// Event listener for button click
submitButton.addEventListener('click', async () => {
    // Get the value of the input field
    const inputData = userEmail.value;
    
    // Check if input is not empty
    if (inputData) {
        try {
            // API URL (replace this with your actual API endpoint)
            const apiURL = 'http://localhost:8080/user/verify-otp?email='+inputData;

            // Prepare the data to send
            const dataToSend = {
                email: inputData, // This will store the user's input
            };

            // Make the POST request using fetch with async/await
            const response = await fetch(apiURL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json; charset=UTF-8',
                },
                body: JSON.stringify(dataToSend), // Convert the data to a JSON string
            });

            // Parse the JSON response
            const result = await response.json();
            // Display the response message
            if(result.status){
                responseMessage.textContent = 'OPT Sent successfully! Response:'+JSON.stringify(result);
            responseMessage.style.color = 'green';

            otpContainer.innerHTML=`
            <input type="text" id="dynamicInput" placeholder="Enter OTP">
                <button id="dynamicSubmit">Submit OTP</button>`;

                const submitButton = document.getElementById('dynamicSubmit');
                const inputField = document.getElementById('dynamicInput');
                
                submitButton.addEventListener('click', async() => {
                    const inputOtp = inputField.value;
                    if (inputOtp) {
                        try {
                            // API URL (replace this with your actual API endpoint)
                            const apiURL = 'http://localhost:8080/user/verify-otp?email='+inputData+'&otp='+inputOtp;
                
                            // Prepare the data to send
                            const dataToSend = {
                                email: inputData,
                                otp:inputOtp, // This will store the user's input
                            };
                
                            // Make the POST request using fetch with async/await
                            const response = await fetch(apiURL, {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json; charset=UTF-8',
                                },
                                body: JSON.stringify(dataToSend), // Convert the data to a JSON string
                            });
                
                            // Parse the JSON response
                            const result = await response.json();
                            if(result.status){
                                // Display the response 
                                responseMessage.textContent = 'OPT Matched';
                                responseMessage.style.color = 'green';
                                window.location.href = 'fileUpload.html';
                            }
                        } catch (error) {
                            // Handle errors (e.g., network issues)
                            responseMessage.textContent = 'Error OTP';
                            responseMessage.style.color = 'red';
                        }
                    } else {
                        // If input is empty, show an error message
                        responseMessage.textContent = 'Please enter some data!';
                        responseMessage.style.color = 'red';
                    }
                });
            }
        } catch (error) {
            // Handle errors (e.g., network issues)
            responseMessage.textContent = 'Error submitting data: ' + error.message;
            responseMessage.style.color = 'red';
        }
    } else {
        // If input is empty, show an error message
        responseMessage.textContent = 'Please enter some data!';
        responseMessage.style.color = 'red';
    }
});
