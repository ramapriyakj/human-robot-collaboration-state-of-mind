Human-Robot Collaboration (HRC) - State of Mind
============

by Ramapriya Kyathanahally Janardhan (janardhan@campus.tu-berlin.de)

Below are the instructions to run the state of mind simulation for HRC.

---

Note:
*   The TestStateofMind.py generates 30 human actions randomly and sends it to tensorflow model for testing.
*   StateOfMInd_TF.py contains the tensorflow code to detect the state of mind.
*   The state of mind corresponding to 30 input actions are dispalyed on the terminal.
*   The code can be integrated to HRC to detect state of mind.

## Running
*	Activating Tensorflow 
    *	tf_source - Tensorflow installation location. eg. ~/Tensorflow/bin
    *	The below commands will activate tensorflow and start morse_predict tensorflow service.

```
source <tf_source>/activate
```

*	Running state of mind code 

```
python TestStateofMind.py 
```




