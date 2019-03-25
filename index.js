import * as tf from '@tensorflow/tfjs';

function imageUploaded(event) {
    const target = event.target;
    const file = target.files[0];

    if (!file) return;

    const imageContainer = document.querySelector("#imageContainer");
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

async function makePrediction(img) {
    const tensorImage = tf.browser.fromPixels(img, 1)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .expandDims();
         
    const predictions = await model.predict(tensorImage).data();

    const firstClass = document.querySelector("#firstClass");
    const secondClass = document.querySelector("#secondClass");

    const ad_percentage = (predictions[0] * 100).toFixed(4);
    const cn_percentage = (predictions[1] * 100).toFixed(4);

    firstClass.innerHTML = `Alzheimer's disease: ${ad_percentage}`;
    secondClass.innerHTML = `Cognitively Normal: ${cn_percentage}`;
}

function ready() {
    const inputFile = document.querySelector("#image");
    inputFile.addEventListener('change', imageUploaded);
}

document.addEventListener("DOMContentLoaded", ready);
let model;

(async function() {
    model = await tf.loadLayersModel('/projects/alzheimer/model/model.json');
})();