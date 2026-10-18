# LinuxONE platform evidence

Evidence that VoxAfrik was built and run on the IBM LinuxONE Community Cloud, captured
from the machine itself. Hostnames are kept; no credentials, addresses or tokens are
included.

## The machine: IBM Z (s390x), machine type 9175

```
$ uname -a
Linux voxafrik 6.8.0-124-generic #124-Ubuntu SMP ... s390x s390x s390x GNU/Linux

$ grep -E "vendor_id|# processors|machine|features" /proc/cpuinfo
vendor_id    : IBM/S390
# processors : 3
machine      : 9175
features     : esan3 zarch stfle msa ldisp eimm dfp edat etf3eh highgprs te vx vxd
               vxe gs vxe2 vxp sort dflt vxp2 nnpa sie
```

Two things matter here. The architecture is `s390x` (IBM Z, not x86), and the CPU feature
list contains `nnpa`, the instruction set of the Telum on-chip AI accelerator. Machine
type 9175 is an IBM z17 / Telum II class system.

![IBM Z host: s390x, machine 9175, nnpa feature, in the LinuxONE JupyterLab terminal](images/ibm-z-s390x.png)

## Containerisation

The work runs inside an IBM-provided JupyterLab container pulled from the LinuxONE
Community Cloud registry.

```
$ docker ps --format "{{.Names}} | {{.Image}} | {{.Status}}"
notebook | registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest | Up 31 hours

$ docker exec notebook uname -a
Linux voxafrik ... s390x GNU/Linux
```

## The Telum accelerator is reachable: zDNN runtime

```
$ ls /usr/local/lib/libzdnn*
/usr/local/lib/libzdnn.a
/usr/local/lib/libzdnn.so
/usr/local/lib/libzdnn.so.0
```

When IBM Granite is loaded through llama.cpp with full offload, it initialises the zDNN
backend and runs on the accelerator, not the CPU:

```
load_tensors:         zDNN model buffer size =   501.47 MiB
ggml_zdnn_init: allocating
ggml_zdnn_init: found 1 device
ggml_zdnn_init: picking default device: zDNN
ggml_zdnn_init: NNPA name: zDNN
zDNN : NNPA = 1 | NNPA_PARMBLKFORMAT_0 = 1 | NNPA_PARMBLKFORMAT_1 = 1 | CPU : LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1
```

The `ggml_zdnn_init: found 1 device` and `NNPA name: zDNN` lines are the confirmation that
the generative layer executes on the Telum accelerator. Reproduce with
[../ibmz/granite_telum.py](../ibmz/granite_telum.py).

![IBM Granite initialising the zDNN backend and running on the Telum accelerator](images/telum-accelerator.png)

## The real-time classifier on the s390x CPU

A real run of `backend/evaluate.py` on the held-out AfriHate test splits, on this machine:

```
language      acc%  macroF1  hate%
Swahili       85.6   0.839   87.4
Igbo          84.6   0.822   82.8
Hausa         73.2   0.685   49.5
Yoruba        76.8   0.579    5.7
Kinyarwanda   67.9   0.650   90.9
------------------------------------
OVERALL       80.0   0.782   74.2
```

Throughput of the full ensemble on the s390x CPU, measured over several passes on 8,000
messages: about 0.22 ms per message, roughly 4,500 messages per second.

## The service is live

```
$ curl -s http://127.0.0.1:7070/health
{"status":"ready"}
```

The API trains the ensemble at start-up and serves the moderation console. A public demo
is exposed through a reverse proxy in front of this instance.
