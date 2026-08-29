# Where Is My Head?

A local p5.js + ml5.js prototype. It uses ml5 BodySegmentation to trace the person's edge with noisy, blurred ASCII characters on a black background. A Teachable Machine image classifier chooses editable text that FaceMesh places in the middle of the detected face. Edit the `CLASS_TEXT` object in `sketch.js` to change the text for each class.

## Run it

Serve this folder with a local web server, then open the local address in a browser and allow camera access. For example:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000` and press `H` to hide/show the small forehead guide.

## Change the image

Replace `assets/crown.svg` with another image, then update this line in `sketch.js` if its filename changes:

```js
crown = loadImage('assets/crown.svg');
```

This prototype runs FaceMesh on the local browser and does not use Roboflow or a cloud API.
