// Variables for neural network
let tfmodel; 
let prediction;
let new_prediction;
let prediction_tensor;
let xs;
// Variable to indicate whether the model is ready to make a new prediction
let ready = 1;
// Dimensions of the canvas
const page_width = 64;
const page_height = 64;
// Variable for the sound file of ambient background noise
let ambient;
// Variables for water starting location and spatial extent
let max_water_len = 8;
let min_water_len = 3;
let water_len;
let water_loc;
// DOM variable
let button1; 

/**
 * Helper function to reshape a 1D array into a desired 2D shape
 * @param orig_array Original array
 * @param rows Desired number of rows in output 2D array
 * @param cols Desired number of columns in output 2D array
 */
function reshape(orig_array, rows, cols) {
    let new_array = [];
    let copy = orig_array.slice(0); // Copy all elements.
  
    for (let r = 0; r < rows; r++) {
      let row = [];
      for (let c = 0; c < cols; c++) {
        var i = r * cols + c;
        if (i < copy.length) {
          row.push(copy[i]);
        }
      }
      new_array.push(row);
    }
    return new_array;
}

// Callback for the "Reset" button once clicked
// The neural network being served was trained on GIFs of waterfalls. Each input to the
// neural network was a binary mask of a GIF frame, with (1) indicating a pixel had
// changed magnitude and (0) indicating a pixel had retained the same magnitude. The
// neural network was tasked with successfully predicting the binary mask of the next
// GIF frame. In this way, the model learned how to replicate the movement of water
// on a pixel-by-pixel level. 
/**
 * Function to initialize the location and spatial extent of water in the image before
 * animation
 */
function reset_xs() {
    // The "xs" variable holds the input to the neural network.
    xs = tf.tidy(() => {
        // Randomly determine the initial starting location of the water and its
        // spatial extent. "water_len" defines the initial horizontal extent of the
        // water and "water_loc" defines the initial vertical starting location of the 
        // water on the canvas.
        let water_len = Math.floor(Math.random() * (max_water_len-min_water_len+1))+min_water_len;
        let water_loc = Math.floor(Math.random() * (page_width-water_len+1));
        // Create an initial binary mask tensor based on "water_len" and "water_loc" 
        let xs_zeros = tf.zeros([page_width,page_height]);
        let temp_row = (arr=[]);
        temp_row.length = page_width;
        temp_row.fill(0);
        for (let ii = 0; ii < water_len; ii++) {
            temp_row[water_loc+ii] = 1;
        }
        let temp = tf.tensor1d(temp_row);
        let xs = tf.add(xs_zeros,temp);
        xs = tf.reshape(xs,[1,page_width,page_height,1]);
        return xs;
    });
    // The "prediction" variable will hold the neural network's predictions for the
    // binary mask of the next frame.
    // Initialize "prediction" to a 2D array of all zeros
    prediction = Array.from(Array(page_height), _ => Array(page_width).fill(0));
}

/**
 * Function to obtain prediction from model
 * Set "ready" variable to 1 once prediction is obtained, indicating that the model
 * is ready to make another prediction
 */
async function model_predict() {
    prediction = tfmodel.predict(xs).dataSync();
    await prediction;
    ready = 1;
}

// Preload the ambient background sound
function preload() {
    ambient = loadSound('http://localhost:8080/watersound.mp3');
}

function setup() {
    //set the canvas size
    createCanvas(64,64);
    // Initialize the location and spatial extent of water in the animation
    reset_xs();
    // Set speed of animation
    frameRate(10);
    // Load the trained neural network
    (async () => {
        console.log('Loading model');
        tfmodel = await tf.loadModel('http://localhost:8080/tfjsmodel/model.json');
        console.log('Model loaded');
    })();
    // "Reset" button
    button1 = createButton('Reset');
    button1.position(80,30);
    button1.mousePressed(function() {
        reset_xs();
    });
    // Loop the ambient background sound
    ambient.loop();
    // DOM variables
    explanationDiv1 = createDiv('AI learned how to animate');
    explanationDiv2 = createDiv('water based on GIFs of');
    explanationDiv3 = createDiv('waterfalls.');
    explanationDiv4 = createDiv('Press the "Reset" button');
    explanationDiv5 = createDiv('to begin a new animation.');
    explanationDiv1.style('font-size', '10px');
    explanationDiv2.style('font-size', '10px');
    explanationDiv3.style('font-size', '10px');
    explanationDiv4.style('font-size', '10px');
    explanationDiv5.style('font-size', '10px');
    explanationDiv1.style('position',10,80); 
    explanationDiv2.style('position',10,90); 
    explanationDiv3.style('position',10,100); 
    explanationDiv4.style('position',10,120); 
    explanationDiv5.style('position',10,130); 
}

function draw() {
    background(0);
    // Return if model is not completely loaded
    if (!tfmodel) {
        return;
    }
    // Obtain a new prediction from the neural network if "ready" is equal to 1
    if (ready==1) {
        model_predict();
    }
    // Reshape the binary mask prediction from a 4-D array into a 2-D 64x64 matrix
    // where each value of a pixel represents the probability that the pixel has changed 
    // value.
    // If the predicted probability for a pixel is greater than 0.5, set the pixel
    // to a blue color.
    new_prediction = reshape(prediction,64,64);
    for (let i = 0; i < 64; i++) {
        for (let j = 0; j < 64; j++) {
            if (new_prediction[i][j] > 0.5) {
                col_red = 0;
                col_green = 255;
                col_blue = 255;
                stroke(col_red,col_green,col_blue);
                point(i,j);
            }
        }
    };
    tf.dispose(prediction);
    tf.dispose(new_prediction);
    // Set the new input to the neural network to be the most reent prediction. In this
    // way, the neural network takes each successive binary mask of a frame and predicts
    // the next one in sequence. 
    prediction_tensor = tf.tensor2d(new_prediction);
    prediction_tensor = tf.reshape(prediction_tensor,[1,64,64,1]);
    xs = prediction_tensor;
    ready = 0;
}
