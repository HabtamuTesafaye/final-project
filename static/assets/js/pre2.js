document.addEventListener('DOMContentLoaded', function () {
    const loading2 = document.getElementById('load2');
    const container = document.getElementById('pred-container');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('chooseFile');
    const showInsightsBtn = document.querySelector('[data-bs-target="#exampleModal"]');

    function setButtonsState(isLoading) {
        if (isLoading) {
            uploadBtn.textContent = 'Loading...';
            uploadBtn.disabled = true;
            showInsightsBtn.disabled = true;
        } else {
            uploadBtn.textContent = 'Upload';
            uploadBtn.disabled = false;
            showInsightsBtn.disabled = false;
        }
    }

    uploadBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Send the file to the server
            let response = await fetch('/upload-csv/', {
                method: 'POST',
                body: formData
            });

            let result = await response.json();
            if (!result.flag) {
                throw new Error(result.error || 'Failed to upload file.');
            }

            // Start the background process for predictions
            await startBackgroundProcess();
        } catch (error) {
            console.error('Error:', error);
            loading2.innerText = 'Failed to upload file and start prediction.';
            loading2.style.display = 'block';  // Ensure loading is visible
        }
    });

    async function startBackgroundProcess() {
        try {
            loading2.style.display = 'block';  // Show loading indicator
            container.style.display = 'none'; // Hide chart container
            setButtonsState(true); 

            let response = await fetch('/start-growth-analysis/');
            let result = await response.json();
            if (!result.flag) {
                throw new Error(result.error || 'Failed to start background process.');
            }

            // Poll for predictions
            await pollForPredictions();
        } catch (error) {
            console.error('Error starting background process:', error);
            loading2.innerText = 'Failed to start background process.';
        }
    }


    async function pollForPredictions() {
        try {
            while (true) {
                let response = await fetch('/get-indicator-future-predictions/');
                let result = await response.json();

                if (!result.flag) {
                    console.log('Background process still running. Polling again...');
                    await new Promise(resolve => setTimeout(resolve, 2000));
                } else {
                    // Hide loading indicator and show chart container
                    loading.style.display = 'none';
                    loading2.style.display = 'none';
                    container.style.display = 'block';
                    setButtonsState(false); 

                    const htmlContent = convertMarkdownToHtml(result.aiInsights);
                    document.getElementById('description').innerHTML = htmlContent

                    // Render the chart with the results
                    await renderChart(result.indicatorFuturePredictions);
                    break;
                }
            }
        } catch (error) {
            console.error('Error fetching future predictions:', error);
            loading2.innerText = 'Failed to fetch future predictions.';
            setButtonsState(false); 
        }
    }

    async function renderChart(indicatorFuturePredictions) {
        try {
            const seriesData = transformDataToSeries(indicatorFuturePredictions);
    
            Highcharts.chart('pred-container', {
                title: {
                    text: 'Future Indicator Predictions',
                    align: 'left'
                },
                subtitle: {
                    text: 'Source: Your Data Source',
                    align: 'left'
                },
                yAxis: {
                    title: {
                        text: 'Predicted Value'
                    },
                    labels: {
                        formatter: function () {
                            return Highcharts.numberFormat(this.value, 2); // Format y-axis values as plain numbers
                        }
                    }
                },
                xAxis: {
                    title: {
                        text: 'Years'
                    },
                    categories: ['3 years', '5 years', '10 years', '15 years', '20 years'], // Add years to x-axis labels
                    accessibility: {
                        rangeDescription: 'Range: 3 to 20 years'
                    }
                },
                legend: {
                    layout: 'vertical',
                    align: 'right',
                    verticalAlign: 'middle'
                },
                plotOptions: {
                    series: {
                        label: {
                            connectorAllowed: false
                        },
                        tooltip: {
                            pointFormatter: function () {
                                return '<b>' + this.series.name + '</b><br/>' +
                                    'Year: ' + this.x + '<br/>' +
                                    'Value: ' + Highcharts.numberFormat(this.y, 2); // Format tooltip values as plain numbers
                            }
                        }
                    }
                },
                series: seriesData,
                responsive: {
                    rules: [{
                        condition: {
                            maxWidth: 500
                        },
                        chartOptions: {
                            legend: {
                                layout: 'horizontal',
                                align: 'center',
                                verticalAlign: 'bottom'
                            }
                        }
                    }]
                }
            });
    
            // Hide loading indicator after rendering
            loading2.style.display = 'none';
        } catch (error) {
            console.error('Error rendering chart:', error);
            loading2.innerText = 'Failed to render chart.'
        }
    }
    

    function transformDataToSeries(indicatorFuturePredictions) {
        const series = [];
        for (const [indicator, data] of Object.entries(indicatorFuturePredictions)) {
            const futureData = data.future_growths.map(fg => [fg.years + ' years', fg.predicted_value]);
            series.push({
                name: indicator,
                data: futureData
            });
        }
        return series;
    }

    function cleanText(text) {
        // Replace Unicode non-breaking spaces with regular spaces
        return text.replace(/\u00a0/g, ' ').trim();
    }
    
    function convertMarkdownToHtml(markdown) {
        let titleAdded = false; // Track if the first title has been processed
    
        // Clean the markdown text first
        markdown = cleanText(markdown);
    
        return markdown
            .replace(/^### (.+)$/gm, '<h3><u>$1</u></h3>')             // Convert ### Header to <h3> and underline
            .replace(/^## (.+)$/gm, function (match, p1) {
                if (!titleAdded) {
                    titleAdded = true;  // Mark that the first title has been processed
                    return `<h2><u>${p1} by AI</u></h2>`;  // Add "by AI" to the first title and underline
                }
                return `<h2><u>${p1}</u></h2>`; // Return other titles as <h2> with underline
            })
            .replace(/^# (.+)$/gm, '<h1><u>$1</u></h1>')              // Convert # Header to <h1> and underline
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')         // Convert **Bold** to <strong>
            .replace(/_(.+?)_/g, '<u>$1</u>')                         // Convert _Underline_ to <u>
            // Convert unordered list items with proper nesting
            .replace(/^\s*[\*\-]\s*(.+)$/gm, function (match, p1) {
                const level = match.match(/^\s*/)[0].length / 2; // Determine nesting level based on leading spaces
                return `${'<ul>'.repeat(level)}<li>${p1}</li>${'</ul>'.repeat(level)}`;
            })
            // Convert ordered list items with proper nesting
            .replace(/^\s*(\d+)\.\s*(.+)$/gm, function (match, p1, p2) {
                const level = match.match(/^\s*/)[0].length / 2; // Determine nesting level based on leading spaces
                return `${'<ol>'.repeat(level)}<li>${p2}</li>${'</ol>'.repeat(level)}`;
            })
            .replace(/<\/ul>\s*<ul>/g, '')  // Remove redundant opening/closing tags between list items
            .replace(/<\/ol>\s*<ol>/g, '')  // Remove redundant opening/closing tags between list items
            .replace(/<\/li>\s*<li>/g, '</li><li>')                   // Ensure proper list item separation
            .replace(/^\n+|\n+$/g, '')                                 // Trim leading and trailing newlines
            .replace(/^(.+?)$/gm, '<p>$1</p>')                        // Convert remaining single lines to paragraphs
            .replace(/<\/p>\n<p>/g, '</p><p>');                       // Remove extra paragraphs between lines
    }
    
    
    startBackgroundProcess()

});
