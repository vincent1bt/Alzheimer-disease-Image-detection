import * as tf from '@tensorflow/tfjs';

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

async function makePrediction(img) {
    const tensorImage = tf.browser.fromPixels(img, 1)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .expandDims();
         
    const predictions = await model.predict(tensorImage).data();
    const ad_percentage = (predictions[0] * 100).toFixed(4);
    const cn_percentage = (predictions[1] * 100).toFixed(4);

    loadingGif.style.display = "none";
    firstClass.innerHTML = `Alzheimer's disease: ${ad_percentage}`;
    secondClass.innerHTML = `Cognitively Normal: ${cn_percentage}`;
}

function ready() {
    firstClass = document.querySelector("#firstClass");
    secondClass = document.querySelector("#secondClass");
    imageContainer = document.querySelector("#imageContainer");
    loadingGif = document.querySelector("#loadingGif");

    const inputFile = document.querySelector("#image");
    inputFile.addEventListener('change', imageUploaded);
}

document.addEventListener("DOMContentLoaded", ready);

let model;
let firstClass;
let secondClass;
let imageContainer;
let loadingGif;

(async function() {
    model = await tf.loadLayersModel('/projects/alzheimer/model/model.json');
})();