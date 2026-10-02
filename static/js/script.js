/**
 * Heart Disease Prediction System - Client-side Controller
 * ========================================================
 * 
 * Beginner-Friendly JavaScript Concepts Explained:
 * ------------------------------------------------
 * 1. Form Submission (event.preventDefault):
 *    By default, submitting an HTML form causes the web browser to refresh the page
 *    or navigate to a new URL. We call `event.preventDefault()` so we can handle
 *    the data asynchronously in JavaScript without any page reload.
 * 
 * 2. Collecting Values & Type Conversion:
 *    HTML input elements always return their values as strings (e.g., "55" or "1.2").
 *    Machine learning models require numerical types. We parse these strings using
 *    `Number()` or `parseFloat()` to convert them to valid JavaScript numbers.
 * 
 * 3. JSON Object Creation:
 *    We construct a standard JavaScript object where keys match the exact 13 feature
 *    names expected by our Flask backend:
 *    [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
 * 
 * 4. Asynchronous fetch() API:
 *    `fetch('/predict', { method: 'POST', ... })` sends an HTTP POST request in the
 *    background. It returns a Promise that resolves when the Flask server responds.
 * 
 * 5. Displaying Results & Error Handling:
 *    We parse the JSON response from Flask. If successful, we update the DOM elements
 *    with prediction outcome, probability meter, and educational disclaimers.
 *    If an error occurs (such as a missing field or server downtime), we catch it
 *    and present a friendly error alert instead of allowing the app to break.
 */

// Define the exact 13 features expected by the trained model
const FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
];

// Curated sample patient records directly from the verified dataset
const SAMPLE_POSITIVE = {
    age: 58,
    sex: 0,
    cp: 0,
    trestbps: 100,
    chol: 248,
    fbs: 0,
    restecg: 0,
    thalach: 122,
    exang: 0,
    oldpeak: 1.0,
    slope: 1,
    ca: 0,
    thal: 2
};

const SAMPLE_NEGATIVE = {
    age: 52,
    sex: 1,
    cp: 0,
    trestbps: 125,
    chol: 212,
    fbs: 0,
    restecg: 1,
    thalach: 168,
    exang: 0,
    oldpeak: 1.0,
    slope: 2,
    ca: 2,
    thal: 3
};

// Wait for the DOM to load before binding event handlers
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = predictBtn.querySelector('.btn-text');
    const resultCard = document.getElementById('resultCard');
    const errorCard = document.getElementById('errorCard');
    const errorMessage = document.getElementById('errorMessage');

    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');
    const resultIcon = document.getElementById('resultIcon');
    const probValueText = document.getElementById('probValueText');
    const probBarFill = document.getElementById('probBarFill');
    const resultDescription = document.getElementById('resultDescription');

    const loadPositiveBtn = document.getElementById('loadPositiveBtn');
    const loadNegativeBtn = document.getElementById('loadNegativeBtn');
    const resetFormBtn = document.getElementById('resetFormBtn');

    /**
     * Helper to populate form fields with sample data
     * @param {Object} sampleData - Dictionary of the 13 feature values
     */
    function populateForm(sampleData) {
        FEATURE_NAMES.forEach(feature => {
            const input = document.getElementById(feature);
            if (input && sampleData[feature] !== undefined) {
                input.value = sampleData[feature];
            }
        });
        hideFeedbackCards();
    }

    /**
     * Hide both result and error feedback cards
     */
    function hideFeedbackCards() {
        resultCard.style.display = 'none';
        errorCard.style.display = 'none';
    }

    /**
     * Display a friendly error message to the user
     * @param {string} msg - The error description
     */
    function showError(msg) {
        resultCard.style.display = 'none';
        errorMessage.textContent = msg;
        errorCard.style.display = 'block';
        errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Attach quick preset button listeners
    if (loadPositiveBtn) {
        loadPositiveBtn.addEventListener('click', () => populateForm(SAMPLE_POSITIVE));
    }
    if (loadNegativeBtn) {
        loadNegativeBtn.addEventListener('click', () => populateForm(SAMPLE_NEGATIVE));
    }
    if (resetFormBtn) {
        resetFormBtn.addEventListener('click', () => {
            form.reset();
            hideFeedbackCards();
        });
    }

    // Main Form Submission Handler
    form.addEventListener('submit', async (event) => {
        // Step 1: Prevent default browser form submission / reload
        event.preventDefault();
        hideFeedbackCards();

        // Step 2: Collect and validate all 13 feature values
        const payload = {};
        const missingFields = [];

        for (const feature of FEATURE_NAMES) {
            const input = document.getElementById(feature);
            if (!input) {
                showError(`Internal configuration error: Missing input for '${feature}'.`);
                return;
            }

            const rawVal = input.value.trim();

            // Client-side validation: ensure value is not empty
            if (rawVal === '') {
                missingFields.push(input.labels && input.labels[0] ? input.labels[0].innerText.split('\n')[0].trim() : feature);
                continue;
            }

            // Convert string to numerical number
            const numVal = Number(rawVal);
            if (isNaN(numVal)) {
                showError(`Invalid numeric value entered for '${feature}'. Please check your input.`);
                input.focus();
                return;
            }

            payload[feature] = numVal;
        }

        // Check if any required fields were left empty
        if (missingFields.length > 0) {
            showError(`Please fill in all required fields. Missing: ${missingFields.join(', ')}.`);
            return;
        }

        // Step 3: Set loading state on the submit button
        predictBtn.disabled = true;
        predictBtn.classList.add('is-loading');
        btnText.textContent = 'Predicting...';

        try {
            // Step 4: Send POST request to Flask /predict API
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            // Parse server response as JSON
            const data = await response.json();

            // Check if Flask returned an HTTP error (e.g., 400 Bad Request or 500)
            if (!response.ok || !data.success) {
                const serverErrMsg = data.error || `Server responded with status code ${response.status}.`;
                showError(serverErrMsg);
                return;
            }

            // Step 5: Render the successful prediction result
            renderPredictionResult(data);

        } catch (networkError) {
            // Catch connection failures, DNS errors, or server unavailable
            console.error('[!] Network or fetch error:', networkError);
            showError('Unable to connect to the prediction server. Please make sure the Flask backend is running at http://127.0.0.1:5000.');
        } finally {
            // Step 6: Restore button state
            predictBtn.disabled = false;
            predictBtn.classList.remove('is-loading');
            btnText.textContent = 'Predict';
        }
    });

    /**
     * Render the model prediction and confidence onto the webpage
     * @param {Object} data - Prediction response from Flask API
     */
    function renderPredictionResult(data) {
        const { prediction, probability } = data;
        const probPercentage = Math.round(probability * 100);

        // Reset classes
        resultCard.classList.remove('result-positive', 'result-negative');

        if (prediction === 1) {
            // Outcome: Class 1 (Positive label according to dataset)
            resultCard.classList.add('result-positive');
            resultIcon.textContent = '⚠️';
            resultTitle.textContent = 'Model Prediction: Positive';
            resultSubtitle.textContent = 'The model predicted class 1 according to the dataset label.';
            resultDescription.textContent = 
                'The trained Logistic Regression model evaluated the 13 clinical measurements and produced a prediction of class 1 based on dataset labels.';
        } else {
            // Outcome: Class 0 (Negative label according to dataset)
            resultCard.classList.add('result-negative');
            resultIcon.textContent = '✓';
            resultTitle.textContent = 'Model Prediction: Negative';
            resultSubtitle.textContent = 'The model predicted class 0 according to the dataset label. This does not prove absence of disease.';
            resultDescription.textContent = 
                'The trained Logistic Regression model evaluated the 13 clinical measurements and produced a prediction of class 0 based on dataset labels. This educational result does not prove absence of disease or medical health.';
        }

        // Set probability readout and animated progress bar
        probValueText.textContent = `${probPercentage}% (${probability})`;
        probBarFill.style.width = `${probPercentage}%`;

        // Reveal the result card and smoothly scroll into view
        resultCard.style.display = 'flex';
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
