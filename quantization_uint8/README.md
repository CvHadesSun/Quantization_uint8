# installation
```buildoutcfg
tensorflow==1.13.1
```
# save tf savedmodel:
```buildoutcfg
python model/test_model.py # first should give the save the model dir.(if dir already exists, there is a error.)
python model_quantization/quantization_uint8.py # save the uint8 model.
```