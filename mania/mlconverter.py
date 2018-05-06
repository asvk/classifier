from coremltools.converters.keras import convert

from keras.models import load_model

model = load_model("uaplates.h5")
coreml_model = convert(model,
                       input_names='image',
                       image_input_names='image',
                       red_bias=-1,
                       blue_bias=-1,
                       green_bias=-1,
                       image_scale=1 / 127.5)
coreml_model.save('uaplates.mlmodel')