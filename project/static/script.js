const lookForwardInput = document.getElementById('look-forward-input');
const timeInput = document.getElementById('time-input');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const timedCheckbox = document.getElementById('timed-chbox');
const graphBox = document.getElementById('graph-box')
const linePlot = document.getElementById('linePlot')
const predicitonPlot = document.getElementById('predictions-plot')
const currentIdxSpan = document.getElementById('currentIdx');
const predictionValueSpan = document.getElementById('predictionValue'); 
const actualValueSpan = document.getElementById('actualValue'); 
const sequenceSpan = document.getElementById('sequence-values');

let currentSequence = 0;
let currentIndex = 0;
let sequences;
let countSequences;
let predictions;
let actual;
let look_back = 10;         // This is defined when traning model
let look_forward;           // This is just to show N sequences ahead on the plot

let autoAdvance;

function draw(direction){
    currentIndex += direction
    if (currentIndex != 0 && currentIndex % look_forward <= 0){
        if(direction == 1){
            if ((currentSequence + look_forward) >= countSequences){
                let diff = countSequences - (currentSequence + look_forward);
                if(diff == 0){
                    currentSequence = 0;
                    currentIndex = 0;
                } else {
                    let offset = look_forward - diff;
                    currentSequence -= offset;
                    currentIndex = offset;
                }
            } else {
                currentIndex = 0;
                currentSequence += look_forward;
            }
        }
        else if(direction == -1){
            currentSequence = ((currentSequence - look_forward) < 0 ? countSequences : currentSequence) - look_forward
            currentIndex = look_forward - 1
        }
    }
    if(currentIndex + currentSequence >= countSequences){
        currentSequence = 0;
        currentIndex = 0;
    }

    let inputSequences = sequences.slice(currentSequence * look_back, (currentSequence + look_forward) * look_back)
    drawLinePlot(inputSequences);
    drawPredictions();
}

timedCheckbox.onchange = function(event) {
    prevBtn.disabled = event.currentTarget.checked;
    nextBtn.disabled = event.currentTarget.checked;

    if (event.currentTarget.checked) {
        autoAdvance = setInterval(() => {
            draw(+1);
        }, parseFloat(timeInput.value) * 1000 || 2000);
    } else {
        clearInterval(autoAdvance);
    }
};

lookForwardInput.onchange = function(event) {
    look_forward = parseInt(event.currentTarget.value);
    if(look_forward <= 0){
        look_forward = 1;
        lookForwardInput.value = 1;
    }
    console.log(look_forward)
    let inputSequences = sequences.slice(currentSequence * look_back, (currentSequence + look_forward) * look_back)
    drawLinePlot(inputSequences);
}

nextBtn.onclick = () => {
    draw(+1);
};

prevBtn.onclick = () => {
    draw(-1);
};


document.addEventListener('DOMContentLoaded', function (){
    look_forward = parseInt(lookForwardInput.value);

    fetch('http://localhost:8080/predict', {method: 'GET',})
    .then(response => response.json())
    .then(data => {
        let arrays = data['sequences']
        let flattenedArrays = arrays.map(subArray => subArray.map(element => element[0]));
        sequences = [].concat(...flattenedArrays);
        countSequences = Math.floor(sequences.length / look_back);
        arrays = data['prediction']
        predictions = [].concat(...arrays)
        arrays = data['actual']
        actual = [].concat(...arrays)
        let inputSequences = sequences.slice(currentSequence * look_back, (currentSequence + look_forward) * look_back)
        drawLinePlot(inputSequences);
        drawPredictions();
    })
    .catch(error => console.error('Error:', error));
})


function drawLinePlot(fullData) {
    const ctx = linePlot.getContext('2d'); 
    const width = linePlot.width
    const height = linePlot.height
    const max = Math.max(...fullData.map(Math.abs));
    const padding = 20; // Vertical padding
    const effectiveHeight = height - 2 * padding;
    ctx.clearRect(0, 0, width, height); 
    ctx.beginPath();
    ctx.strokeStyle = 'gray'; 
    ctx.lineWidth = 1; 

    function calculateY(value) {
        return padding + effectiveHeight / 2 * (1 - (value / max));
    }

    // Drawing the full look_forward * look_back number sequence as a background
    fullData.forEach((value, index) => {
        const x = index * ((width - 20) / fullData.length); // Adjust to fit 50 values
        const y = calculateY(value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Highlight the current 10 number sequence
    ctx.beginPath();
    ctx.strokeStyle = 'red'; 
    ctx.lineWidth = 2; 

    let last_x, last_y;
    fullData.slice(currentIndex * look_back, (currentIndex + 1) * look_back ).forEach((value, index) => {
        const x = (index + (currentIndex * look_back)) * ((width - 20) / fullData.length);
        const y = calculateY(value);
        last_x = x 
        last_y = y
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw a line fron N to N+1 with the Predicted value
    ctx.beginPath();
    ctx.moveTo(last_x, last_y);
    ctx.strokeStyle = 'blue'; 
    ctx.lineWidth = 1; 

    let x = (10 + (currentIndex * look_back)) * ((width - 20) / fullData.length);
    let y = calculateY(predictions[currentIndex + currentSequence])
    ctx.lineTo(x, y);
    ctx.stroke();

    // Draw a line fron N to N+1 with the Actual value
    ctx.beginPath();
    ctx.moveTo(last_x, last_y);
    ctx.strokeStyle = 'green'; 
    ctx.lineWidth = 1; 

    x = (10 + (currentIndex * look_back)) * ((width - 20) / fullData.length);
    y = calculateY(actual[currentIndex + currentSequence])
    ctx.lineTo(x, y);
    ctx.stroke();

    // Draw a horizontal line at y = 0
    ctx.beginPath();
    ctx.strokeStyle = 'black'; 
    ctx.lineWidth = 0.5; 
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Update the text spans with the order number of sequence, predicted and actual value
    currentIdxSpan.textContent = (currentSequence + currentIndex) + " / " + countSequences;
    predictionValueSpan.textContent = predictions[currentIndex + currentSequence].toFixed(4); 
    actualValueSpan.textContent = actual[currentIndex + currentSequence].toFixed(4); 

    // Update the text spanws with all the elements of the sequence
    const sequence = fullData.slice(currentIndex * look_back, (currentIndex + 1) * look_back );
    const formattedSequence = sequence.map(number => number.toFixed(2))
    sequenceSpan.textContent = formattedSequence.join(', ');
}

function drawPredictions(){
    const ctx = predicitonPlot.getContext('2d'); 
    const width = predicitonPlot.width
    const height = predicitonPlot.height
    const max = Math.max(...predictions.map(Math.abs));
    const padding = 20; // Vertical padding
    const effectiveHeight = height - 2 * padding;

    ctx.clearRect(0, 0, width, height);
    ctx.beginPath();
    ctx.strokeStyle = 'blue'; 
    ctx.lineWidth = 1; 

    function calculateY(value) {
        return padding + effectiveHeight / 2 * (1 - (value / max));
    }

    // Draw and fill a line of Predicted values
    predictions.forEach((value, index) => {
        const x = index * (width / (predictions.length));
        const y = calculateY(value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.lineTo(width, height / 2);
    ctx.lineTo(0, height / 2);
    ctx.closePath();

    ctx.fillStyle = 'rgba(0, 0, 255, 0.3)'; 
    ctx.globalCompositeOperation = 'source-over';
    ctx.fill();
    ctx.stroke();
    ctx.globalCompositeOperation = 'destination-over';

    // Draw and fill a line of Actual values
    ctx.beginPath();
    ctx.strokeStyle = 'green'; // Color for the background sequence
    ctx.lineWidth = 1; // Thinner line for the background

    actual.forEach((value, index) => {
        const x = index * (width / actual.length); // Adjust to fit 50 values
        const y = calculateY(value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    
    ctx.lineTo(width, height / 2);
    ctx.lineTo(0, height / 2);
    ctx.closePath();

    ctx.fillStyle = 'rgba(0, 255, 0, 0.3)'; 
    ctx.fill();
    ctx.stroke();

    // Draw a horizontal line at y = 0
    ctx.beginPath();
    ctx.strokeStyle = 'black'; 
    ctx.lineWidth = 0.3; 
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw a red rectangle as a current sequence indicator on the plot
    ctx.fillStyle = "red";
    ctx.fillRect((currentIndex + currentSequence) * (width / actual.length), (height / 2) - 2 ,4,4); // fill in the pixel at (10,10
}


