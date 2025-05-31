document.addEventListener('DOMContentLoaded', () => {
    function generateDataset() {
        const form = document.getElementById('geneForm');
        const formData = new FormData(form);
        const userInput = {};
        const chromosomeColumns = {};
        const geneticCategoryColumns = {};

        // Initialize chromosome columns
        for (let i = 2; i <= 22; i++) {
            chromosomeColumns[`chromosome_${i}`] = 0;
        }
        chromosomeColumns["chromosome_X"] = 0;
        chromosomeColumns["chromosome_X,Y"] = 0;
        chromosomeColumns["chromosome_Y"] = 0;

        // Initialize genetic category columns
        const geneticCategories = [
            "genetic-category_Genetic Association",
            "genetic-category_Genetic Association, Functional",
            "genetic-category_Rare Single Gene Mutation",
            "genetic-category_Rare Single Gene Mutation, Functional",
            "genetic-category_Rare Single Gene Mutation, Genetic Association",
            "genetic-category_Rare Single Gene Mutation, Genetic Association, Functional",
            "genetic-category_Rare Single Gene Mutation, Syndromic",
            "genetic-category_Rare Single Gene Mutation, Syndromic, Functional",
            "genetic-category_Rare Single Gene Mutation, Syndromic, Genetic Association",
            "genetic-category_Rare Single Gene Mutation, Syndromic, Genetic Association, Functional",
            "genetic-category_Syndromic",
            "genetic-category_Syndromic, Genetic Association"
        ];

        geneticCategories.forEach(category => {
            geneticCategoryColumns[category] = 0;
        });

        // Process form data
        for (const [key, value] of formData.entries()) {
            if (key === 'chromosome') {
                Object.keys(chromosomeColumns).forEach(col => {
                    chromosomeColumns[col] = col === value ? 1 : 0;
                });
            } else if (key === 'genetic-category') {
                Object.keys(geneticCategoryColumns).forEach(col => {
                    geneticCategoryColumns[col] = col === value ? 1 : 0;
                });
            } else {
                userInput[key] = isNaN(value) ? value : Number(value);
            }
        }

        // Add "syndromic" column with value 0
        userInput["syndromic"] = 0;

        // Reordered keys array as per your requirement
        const orderedKeys = [
            "gene-score",
            "syndromic",
            "number-of-reports",
            "chromosome_10",
            "chromosome_11",
            "chromosome_12",
            "chromosome_13",
            "chromosome_14",
            "chromosome_15",
            "chromosome_16",
            "chromosome_17",
            "chromosome_18",
            "chromosome_19",
            "chromosome_2",
            "chromosome_20",
            "chromosome_21",
            "chromosome_22",
            "chromosome_3",
            "chromosome_4",
            "chromosome_5",
            "chromosome_6",
            "chromosome_7",
            "chromosome_8",
            "chromosome_9",
            "chromosome_X",
            "chromosome_X,Y",
            "chromosome_Y",
            "genetic-category_Genetic Association",
            "genetic-category_Genetic Association, Functional",
            "genetic-category_Rare Single Gene Mutation",
            "genetic-category_Rare Single Gene Mutation, Functional",
            "genetic-category_Rare Single Gene Mutation, Genetic Association",
            "genetic-category_Rare Single Gene Mutation, Genetic Association, Functional",
            "genetic-category_Rare Single Gene Mutation, Syndromic",
            "genetic-category_Rare Single Gene Mutation, Syndromic, Functional",
            "genetic-category_Rare Single Gene Mutation, Syndromic, Genetic Association",
            "genetic-category_Rare Single Gene Mutation, Syndromic, Genetic Association, Functional",
            "genetic-category_Syndromic",
            "genetic-category_Syndromic, Genetic Association"

        ];
        
        // Create the dataset with ordered keys
        const dataset = [orderedKeys.reduce((acc, key) => {
            acc[key] = key in userInput ? userInput[key] : key in chromosomeColumns ? chromosomeColumns[key] : geneticCategoryColumns[key];
            return acc;
        }, {})];

        // Send the dataset to the Flask app
        sendToFlaskApp({ dataset });
    }

    function sendToFlaskApp(payload) {
        fetch('https://ubiquitous-yodel-rq65vg49jjpcxrv-5000.app.github.dev/predict_gene', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),  // Send the entire payload, which includes the dataset
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('output').textContent = `Error: ${error}`;
        });
    }
    

    const submitButton = document.getElementById('submitButton');
    if (submitButton) {
        submitButton.addEventListener('click', generateDataset);
    } else {
        console.error('Button with ID "submitButton" not found.');
    }
});
