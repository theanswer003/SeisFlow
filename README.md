Geostatistical seismic inversion with TensorFlow
==================
Accelerating geostatistical seismic inversion using the distributed TensorFlow with GPUs.

**Setup**
* TensorFlow, version r1.0 or later
```
pip install tensorflow        (CPU version)
pip install tensorflow-gpu    (GPU version)
```

**Usage**
* Setting parameters in `conf/setting.py` to configure computing devices, geostatistical simulation parameters, etc.
* After configuration, you can run the `bin/start.py`:
```
python bin/start.py
```
```

**Dataset**
* Because of data confidentiality, the full dataset are not provided herein, but I provide a small dataset (100 * 100 * 200)  in the folder `db` for test and play with the code.

**Contact**
* Feel free to email mingliangliu[at]outlook[dot]com for any pertinent questions/bugs regarding the code.