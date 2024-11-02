# Alzheimer's disease Image detection

This app uses TensorflowJS.

You can see this app running [here](https://vincentblog.link/portfolio/alzheimer)
and a blog post about the app [here](https://vincentblog.link/posts/detecting-alzheimer-s-desease-with-deep-learning).

If you want to run this app you need a simple server:

```
python -m http.server 8000
```

First you have to change the route of the model:

```
tf.loadLayersModel('/assets/model/model.json');
```

and the route of the sample image:

```
const url = "./assets/img.jpg";
```

Finally run webpack:

```
npm run build
```
