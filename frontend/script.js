const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");

let drawing = false;


// Canvas setup

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 12;
ctx.lineCap = "round";
ctx.lineJoin = "round";


// Get mouse position

function getMousePosition(event) {

    const rect = canvas.getBoundingClientRect();

    return {
        x: (event.clientX - rect.left) *
            (canvas.width / rect.width),

        y: (event.clientY - rect.top) *
            (canvas.height / rect.height)
    };
}


// Start drawing

canvas.addEventListener("mousedown", (event) => {

    drawing = true;

    const position = getMousePosition(event);

    ctx.beginPath();

    ctx.moveTo(
        position.x,
        position.y
    );
});


// Draw

canvas.addEventListener("mousemove", (event) => {

    if (!drawing) {
        return;
    }

    const position = getMousePosition(event);

    ctx.lineTo(
        position.x,
        position.y
    );

    ctx.stroke();
});


// Stop drawing

canvas.addEventListener("mouseup", () => {
    drawing = false;
});

canvas.addEventListener("mouseleave", () => {
    drawing = false;
});


// Clear button

clearBtn.addEventListener("click", () => {

    ctx.fillStyle = "black";

    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.strokeStyle = "white";

    prediction.textContent = "?";

    confidence.textContent =
        "Draw a digit to get started.";
});


// Predict button

predictBtn.addEventListener("click", async () => {

    const imageData = ctx.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
    );

    const data = imageData.data;


    // Find digit boundaries

    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = -1;
    let maxY = -1;


    for (let y = 0; y < canvas.height; y++) {

        for (let x = 0; x < canvas.width; x++) {

            const index =
                (y * canvas.width + x) * 4;

            const value = data[index];

            if (value > 20) {

                minX = Math.min(minX, x);
                minY = Math.min(minY, y);

                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }


    // Nothing drawn

    if (maxX === -1) {

        prediction.textContent = "?";

        confidence.textContent =
            "Please draw a digit first.";

        return;
    }


    // Add padding

    const padding = 15;

    minX = Math.max(
        0,
        minX - padding
    );

    minY = Math.max(
        0,
        minY - padding
    );

    maxX = Math.min(
        canvas.width - 1,
        maxX + padding
    );

    maxY = Math.min(
        canvas.height - 1,
        maxY + padding
    );


    const width =
        maxX - minX + 1;

    const height =
        maxY - minY + 1;


    // Preserve digit proportions

    const targetSize = 20;

    const scale =
        Math.min(
            targetSize / width,
            targetSize / height
        );

    const newWidth =
        width * scale;

    const newHeight =
        height * scale;


    // Create 28x28 MNIST image

    const smallCanvas =
        document.createElement("canvas");

    smallCanvas.width = 28;
    smallCanvas.height = 28;

    const smallCtx =
        smallCanvas.getContext("2d");


    smallCtx.fillStyle = "black";

    smallCtx.fillRect(
        0,
        0,
        28,
        28
    );


    // Center digit

    const offsetX =
        (28 - newWidth) / 2;

    const offsetY =
        (28 - newHeight) / 2;


    smallCtx.drawImage(
        canvas,

        minX,
        minY,
        width,
        height,

        offsetX,
        offsetY,
        newWidth,
        newHeight
    );


    // Get pixels

    const smallImage =
        smallCtx.getImageData(
            0,
            0,
            28,
            28
        );


    const pixels = [];


    for (
        let i = 0;
        i < smallImage.data.length;
        i += 4
    ) {

        pixels.push(
            smallImage.data[i]
        );
    }


    // Loading state

    prediction.textContent = "...";

    confidence.textContent =
        "Analyzing...";


    try {

        const response = await fetch(
            "https://neural-network-from-scratch-630s.onrender.com/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    pixels: pixels
                })
            }
        );


        if (!response.ok) {
            throw new Error("Server error");
        }


        const result =
            await response.json();


        // Prediction

        const predictedDigit =
            result.prediction;


        // Probability of predicted digit

        const predictedConfidence =
            result.probabilities[
                predictedDigit
            ] * 100;


        prediction.textContent =
            predictedDigit;


        confidence.textContent =
            `${predictedConfidence.toFixed(1)}% confidence`;


    } catch (error) {

        console.error(error);

        prediction.textContent = "!";

        confidence.textContent =
            "Could not connect to the backend.";
    }
});