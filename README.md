# Alzheimer's disease Image detection

This app uses TensorflowJS.

You can see this app running [here](https://vincentblog.xyz/portfolio/alzheimer)
and a blog post about the app in spanish [here](https://vincentblog.xyz/posts/detectando-alzheimer-con-deep-learning).

If you want to run this app you need a simple server:

```
python -m http.server 8000
```

And change the route of the model:

```
tf.loadLayersModel('http://0.0.0.0:8000/assets/model/model.json');
```