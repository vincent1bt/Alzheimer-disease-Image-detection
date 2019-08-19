import * as tf from '@tensorflow/tfjs';
import jet from "./colormaps";

function imageUploaded(event) {
    const target = event.target;
    const file = target.files[0];

    if (!file) return;

    firstClass.innerHTML = "";
    secondClass.innerHTML = "";
    loadingGif.style.display = "block";
    imageContainer.src = "";

    const imageUrl = window.webkitURL.createObjectURL(file);
    imageContainer.src = imageUrl;

    let reader = new FileReader();

    reader.onload = readerEvent => {
        let img = document.createElement('img');
        img.src = readerEvent.target.result;
        img.width = 128;
        img.height = 128;
        img.onload = () => makePrediction(img);
    };

    reader.readAsDataURL(file);
}

function makePrediction(img) {
    const [batched_image, resized_image] = normalizeImage(img);

    predict_and_print_maps(batched_image, resized_image);
}

function normalizeImage(img) {
    return tf.tidy(() => {
        const tensorImage = tf.browser.fromPixels(img, 1).toFloat();
        const resized_image = tensorImage.resizeNearestNeighbor([128, 128]);
    
        const offset = tf.scalar(127.5);
        const normalized_image = resized_image.div(offset).sub(tf.scalar(1.0));
    
        const batched_image = normalized_image.expandDims();
        return [batched_image, resized_image];
    });
}

function predict_and_print_maps(batched_image, resized_image) {
    let class_weights = model.getLayer("dense").getWeights();
    const final_conv_layer = model.getLayer("Conv_1");

    const custom_model = tf.model({inputs: model.getLayer("input_1").input, outputs: [final_conv_layer.output, model.getLayer("dense").output]});
    
    let [conv_outputs, predictions] = custom_model.predict(batched_image);

    // conv_outputs = conv_outputs[0, :, :, :];
    const conv_outputs_shape = conv_outputs.shape.slice(1, 3);
    conv_outputs = conv_outputs.arraySync()[0];

    let cam = tf.zeros(conv_outputs_shape, "float32");

    predictions = predictions.arraySync()[0];

    const ad_percentage = (predictions[0] * 100).toFixed(4);
    const cn_percentage = (predictions[1] * 100).toFixed(4);

    firstClass.innerHTML = `Alzheimer's disease: ${ad_percentage}`;
    secondClass.innerHTML = `Cognitively Normal: ${cn_percentage}`;

    loadingGif.style.display = "none";

    const image_class = ad_percentage > cn_percentage ? 0 : 1;

    class_weights = class_weights[0].arraySync();
    //class_weights[:, image_class]
    const image_class_weights = class_weights.map(class_weight => {
        return class_weight[image_class];
    });

    image_class_weights.forEach(function (weight, index) {
        //conv_outputs[:, :, index];
        let conv = conv_outputs.map(first_conv_output => {
            return first_conv_output.map(second_conv_output => {
                return second_conv_output[index];
            });
        });

        cam = cam.add(tf.tensor(conv).mul(tf.scalar(weight)));
    });

    let max_value = cam.max().dataSync()[0];
    cam = cam.div(tf.scalar(max_value));

    let cam_3d = tf.stack([cam, cam, cam], 2);
    cam_3d = tf.image.resizeBilinear(cam_3d, [128, 128]);

    cam_3d = cam_3d.mul(tf.scalar(255)).toInt();

    let cam_3d_flat = cam_3d.reshape([128 * 128 * 3]);

    let heatmap = makeColor(cam_3d_flat.arraySync());

    let squezed_image = resized_image.squeeze();

    let resized_image_3d = tf.stack([squezed_image, squezed_image, squezed_image], 2);

    heatmap = heatmap.map(value => {
        if (value < 20) {
            return 0;
        }
        return value;
    });

    heatmap = tf.tensor(heatmap, [128, 128, 3]);

    heatmap = heatmap.mul(tf.scalar(0.5)).add(resized_image_3d);

    let heatmap_max_value = heatmap.max().dataSync()[0];
    heatmap = heatmap.div(tf.scalar(heatmap_max_value));

    heatmap = tf.image.resizeBilinear(heatmap, [200, 200]);

    tf.browser.toPixels(heatmap, activationCanvas);
}

//enforceBounds, interpolateLinearly and makeColor are functions from
// https://github.com/mlmed/dl-web-xray
// used to apply the colormap to the activation maps

function enforceBounds(pixel) {
    if (pixel < 0) {
        return 0;
    } else if (pixel > 1){
        return 1;
    } else {
        return pixel;
    }
}
function interpolateLinearly(pixel, values) {
    let x_values = [];
    let r_values = [];
    let g_values = [];
    let b_values = [];
    
    for (let index in values) {
        x_values.push(values[index][0]);
        r_values.push(values[index][1][0]);
        g_values.push(values[index][1][1]);
        b_values.push(values[index][1][2]);
    }

    let index = 1;

    while (x_values[index] < pixel) {
        index = index + 1;
    }

    index = index - 1;

    let width = Math.abs(x_values[index] - x_values[index + 1]);
    let scaling_factor = (pixel - x_values[index]) / width;

    let r = r_values[index] + scaling_factor * (r_values[index +1] - r_values[index])
    let g = g_values[index] + scaling_factor * (g_values[index +1] - g_values[index])
    let b = b_values[index] + scaling_factor * (b_values[index +1] - b_values[index])
    
    return [enforceBounds(r), enforceBounds(g), enforceBounds(b)];
}

function makeColor(data) {
    for (let index = 0; index < data.length; index += 3) {
        let color = interpolateLinearly(data[index] / 255, jet);
        
        data[index] = Math.round(255 * color[0]); 
        data[index + 1] = Math.round(255 * color[1]);
        data[index + 2] = Math.round(255 * color[2]);
    }

    return data;
}

function ready() {
    firstClass = document.querySelector("#firstClass");
    secondClass = document.querySelector("#secondClass");
    imageContainer = document.querySelector("#imageContainer");
    loadingGif = document.querySelector("#loadingGif");

    activationCanvas = document.querySelector("#activation_canvas");

    const inputFile = document.querySelector("#image");
    inputFile.addEventListener('change', imageUploaded);

    const example_button = document.querySelector("#example_button");
    example_button.addEventListener('click', runExample);
}

function runExample() {
    loadingGif.style.display = "block";
    const url = "./assets/img.jpg";
    let img = new Image(200, 200);
    img.src = url;

    imageContainer.src = url;

    img.onload = () => makePrediction(img);
}

document.addEventListener("DOMContentLoaded", ready);

let model;
let firstClass;
let secondClass;
let imageContainer;
let loadingGif;
let activationCanvas;

(async function() {
    model = await tf.loadLayersModel('/projects/alzheimer/model/model.json');
})();