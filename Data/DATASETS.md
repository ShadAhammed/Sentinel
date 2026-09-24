# SENTINEL-X Dataset Catalog

## CNN-Training

**10 classes.** These are the labels in `Data/CNN-Data` after the 23 September 2026 MV addition. Training counts are the local `train` and `validation` crops combined. One MV source photo was added (the train copy when it existed). Armoured personnel carriers and tanks went into `military_vehicle`. MV soldiers went into `camouflage_soldier`. Crops with a short side under 32 pixels were not added (345 refused).

| Label | Training instances | Test instances |
|---|---:|---:|
| Artillery | 801 | 50 |
| M. Rocket Launcher | 341 | 45 |
| Missile | 411 | 36 |
| Radar | 253 | 34 |
| Soldier | 1,000 | 50 |
| camouflage_soldier | 1,109 | 58 |
| military_aircraft | 1,000 | 50 |
| military_vehicle | 3,689 | 187 |
| military_warship | 900 | 100 |
| trench | 37 | 7 |
| **Total** | **9,541** | **617** |

---

This file describes the object-detection datasets in `Data/YOLO`. KIIT-MiTA and military_object_dataset counts were computed from the files on disk (21 September 2026). SULAND v2 counts were computed on 22 September 2026. MV counts were recomputed on 23 September 2026 after air-fighter and bomber were removed. They are not copied from vendor marketing pages.

| Dataset | Task | Format | Classes | Images | Label files | Instances | Disk |
|---|---|---|---:|---:|---:|---:|---:|
| KIIT-MiTA | Military / tactical assets | YOLO | 7 | 1,700 | 1,700 | 4,586 | 125.3 MB |
| military_object_dataset | Military + civilian objects | YOLO | 12 | 26,315 | 26,315 | 50,822 | 3,991.6 MB |
| SULAND_v2 | Surface landmine detection | YOLO + COCO JSON | 2 | 33,771 | 33,771 | 12,427 | 15.5 GB |
| MV | Military vehicle recognition | YOLO | 3 | 2,702 | 2,702 | 4,238 | 152.1 MB |

**Total:** 64,488 images, 72,073 annotated instances, ~19.8 GB. The MV disk figure is from before the aircraft photos were removed.

### Training instances only

| Dataset | Label | Instances |
|---|---|---:|
| KIIT-MiTA | Artilary | 238 |
| KIIT-MiTA | Missile | 367 |
| KIIT-MiTA | Radar | 223 |
| KIIT-MiTA | M. Rocket Launcher | 303 |
| KIIT-MiTA | Soldier | 923 |
| KIIT-MiTA | Tank | 683 |
| KIIT-MiTA | Vehicle | 974 |
| military_object_dataset | camouflage_soldier | 4,477 |
| military_object_dataset | weapon | 1,210 |
| military_object_dataset | military_tank | 17,454 |
| military_object_dataset | military_truck | 1,245 |
| military_object_dataset | military_vehicle | 1,963 |
| military_object_dataset | civilian | 52 |
| military_object_dataset | soldier | 6,502 |
| military_object_dataset | civilian_vehicle | 519 |
| military_object_dataset | military_artillery | 439 |
| military_object_dataset | trench | 4 |
| military_object_dataset | military_aircraft | 7,337 |
| military_object_dataset | military_warship | 2,134 |

Train-set instance totals: KIIT-MiTA **3,711** · military_object_dataset **43,336**.

---

## 1. KIIT-MiTA

**Path:** `Data/YOLO/KIIT-MiTA`  
**Config:** `data.yaml`  
**Run:** `run_yolo.py` in this folder. Change the model name in that file, then run `python run_yolo.py`.  
**Source style:** YOLO detection (one `.jpeg` + one `.txt` per image)  
**Annotation:** `class_id x_center y_center width height` (normalized 0–1)

### Labels

| ID | Class name (as in YAML) |
|---:|---|
| 0 | Artilary |
| 1 | Missile |
| 2 | Radar |
| 3 | M. Rocket Launcher |
| 4 | Soldier |
| 5 | Tank |
| 6 | Vehicle |

`nc: 7` in `data.yaml`. The spelling **Artilary** is kept as stored in the config.

### Train / test / val size

Folders on disk are `train`, `test`, and `valid` (not `val`).

| Split | Folder | Images | Label files | Instances | Disk |
|---|---|---:|---:|---:|---:|
| Train | `train/` | 1,360 | 1,360 | 3,711 | 99.4 MB |
| Test | `test/` | 170 | 170 | 419 | 13.0 MB |
| Validation | `valid/` | 170 | 170 | 456 | 13.0 MB |
| **Total** | | **1,700** | **1,700** | **4,586** | **125.3 MB** |

Split ratio: **80.0% / 10.0% / 10.0%**. Every image has a non-empty label file.

`data.yaml` points validation at `valid/images` and test at `test/images`. Use the folder table above when reporting split sizes.

### Instances per class

| ID | Class | Train | Test | Valid | Total |
|---:|---|---:|---:|---:|---:|
| 0 | Artilary | 238 | 40 | 26 | 304 |
| 1 | Missile | 367 | 36 | 44 | 447 |
| 2 | Radar | 223 | 34 | 30 | 287 |
| 3 | M. Rocket Launcher | 303 | 45 | 38 | 386 |
| 4 | Soldier | 923 | 120 | 108 | 1,151 |
| 5 | Tank | 683 | 58 | 94 | 835 |
| 6 | Vehicle | 974 | 86 | 116 | 1,176 |
| | **All** | **3,711** | **419** | **456** | **4,586** |

Most frequent: Vehicle and Soldier. Least frequent: Radar and Artilary.

---

## 2. military_object_dataset

**Path:** `Data/YOLO/military_object_dataset`  
**Config:** `data.yaml`  
**Run:** `run_yolo.py` in this folder. Change the model name in that file, then run `python run_yolo.py`.  
**Source style:** YOLOv8 detection (one `.jpg` + one `.txt` per image)  
**Annotation:** `class_id x_center y_center width height` (normalized 0–1)

### Labels

| ID | Class | Description |
|---:|---|---|
| 0 | camouflage_soldier | Soldiers in camouflage |
| 1 | weapon | Handheld firearms and similar weapons |
| 2 | military_tank | Armored combat vehicles |
| 3 | military_truck | Troop / supply trucks |
| 4 | military_vehicle | Other military vehicles |
| 5 | civilian | Unarmed non-military people |
| 6 | soldier | Military personnel without camouflage |
| 7 | civilian_vehicle | Non-military cars and trucks |
| 8 | military_artillery | Large-caliber / heavy guns |
| 9 | trench | Defensive earthworks |
| 10 | military_aircraft | Combat, surveillance, or transport aircraft |
| 11 | military_warship | Naval combat vessels |

### Train / test / val size

| Split | Folder | Images | Label files | Empty labels | Instances | Disk |
|---|---|---:|---:|---:|---:|---:|
| Train | `train/` | 21,978 | 21,978 | 294 | 43,336 | 3,323.5 MB |
| Validation | `val/` | 2,941 | 2,941 | 273 | 5,081 | 498.4 MB |
| Test | `test/` | 1,396 | 1,396 | 127 | 2,405 | 169.6 MB |
| **Total** | | **26,315** | **26,315** | **694** | **50,822** | **3,991.6 MB** |

Split ratio: **83.5% / 11.2% / 5.3%**.  
694 label files exist but contain no boxes (negative / empty images).

### Instances per class

| ID | Class | Train | Val | Test | Total |
|---:|---|---:|---:|---:|---:|
| 0 | camouflage_soldier | 4,477 | 510 | 389 | 5,376 |
| 1 | weapon | 1,210 | 358 | 0 | 1,568 |
| 2 | military_tank | 17,454 | 1,787 | 818 | 20,059 |
| 3 | military_truck | 1,245 | 148 | 96 | 1,489 |
| 4 | military_vehicle | 1,963 | 307 | 194 | 2,464 |
| 5 | civilian | 52 | 1 | 0 | 53 |
| 6 | soldier | 6,502 | 745 | 560 | 7,807 |
| 7 | civilian_vehicle | 519 | 42 | 25 | 586 |
| 8 | military_artillery | 439 | 117 | 50 | 606 |
| 9 | trench | 4 | 3 | 37 | 44 |
| 10 | military_aircraft | 7,337 | 1,063 | 236 | 8,636 |
| 11 | military_warship | 2,134 | 0 | 0 | 2,134 |
| | **All** | **43,336** | **5,081** | **2,405** | **50,822** |

Class imbalance is severe:

- **military_tank** is ~39% of all boxes.
- **civilian** (53) and **trench** (44) are rare.
- **military_warship** appears only in train (none in val/test).
- **weapon** and **civilian** have **0** test instances.
- Most **trench** boxes are in test (37 of 44), not train.

`data.yaml` points validation at `val/images` and test at `test/images`.

---

## 3. SULAND v2

**Path:** `Data/YOLO/SULAND_v2`  
**Configs:** `data-iid/ITA.yaml`, `data-ood/USA.yaml`  
**Source:** Hugging Face `SagarLekhak/SULAND_v2_RGB_Surface_Landmine_Dataset` (refined annotations on the original SULAND images)  
**Source style:** YOLO text labels plus COCO `instances_*.json`  
**Annotation:** `class_id x_center y_center width height` (normalized 0-1)

IID is Italy (`data-iid`). OOD is USA (`data-ood`). OOD has a validation split only.

### Labels

| ID | Class name |
|---:|---|
| 0 | butterfly |
| 1 | starfish |

### Images

| Split | IID (Italy) | OOD (USA) |
|---|---:|---:|
| Train | 22,756 | 0 |
| Validation | 2,836 | 4,436 |
| Test | 3,743 | 0 |
| **Total** | **29,335** | **4,436** |

**33,771** images in total. Every label file has a matching `.jpg`. Many images have no box.

### Instances per class

| ID | Class | IID train | IID val | IID test | OOD val | Total |
|---:|---|---:|---:|---:|---:|---:|
| 0 | butterfly | 3,430 | 431 | 539 | 2,454 | 6,854 |
| 1 | starfish | 3,281 | 373 | 517 | 1,402 | 5,573 |
| | **All** | **6,711** | **804** | **1,056** | **3,856** | **12,427** |

The YAML `path` fields still say `./datasets/SULAND_v2/...`. Local training should use `Data/YOLO/SULAND_v2`.

---

## 4. MV

**Path:** `Data/YOLO/MV`  
**Config:** `data.yaml`  
**Run:** `run_yolo.py` in this folder. Change the model name in that file, then run `python run_yolo.py`.  
**Source:** Roboflow project `military-vehicle-recognition` version 7, CC BY 4.0  
**Source style:** YOLO detection (one `.jpg` + one `.txt` per image)  
**Annotation:** `class_id x_center y_center width height` (normalized 0-1)

`data.yaml` points validation at `valid/images` and test at `test/images`.

### Labels

| ID | Class name |
|---:|---|
| 0 | armoured personnel carrier |
| 1 | soldier |
| 2 | tank |

Air-fighter and bomber were removed on 23 September 2026. Photos that contained only those boxes were deleted. Mixed photos kept their other boxes. Class ids were packed to 0, 1, and 2.

### Train / valid / test size

| Split | Folder | Images | Label files | Instances |
|---|---|---:|---:|---:|
| Train | `train/` | 2,361 | 2,361 | 3,684 |
| Validation | `valid/` | 227 | 227 | 366 |
| Test | `test/` | 114 | 114 | 188 |
| **Total** | | **2,702** | **2,702** | **4,238** |

33 training label files and 3 validation label files are empty. Those images were already empty and were left in place.

These 2,702 files come from **735** source photos. Every filename contains `.rf.`, which is Roboflow's generated-copy marker. Distinct source names: 599 in train, 215 in valid, and 111 in test.

Keeping one file per source (the train file when it exists) leaves these labels. 91 photos have two or more classes, 634 have one class, and 10 have no boxes.

| Class | Photos | Boxes on the kept file | Photos with only this class |
|---|---:|---:|---:|
| armoured personnel carrier | 309 | 431 | 219 |
| soldier | 124 | 312 | 108 |
| tank | 384 | 445 | 307 |

119 sources are in train and valid, 57 are in train and test, and 14 are in valid and test. **57 of the 114 test files** are copies of a photo that is also in train.

### Instances per class

| ID | Class | Train | Valid | Test | Total |
|---:|---|---:|---:|---:|---:|
| 0 | armoured personnel carrier | 1,296 | 131 | 71 | 1,498 |
| 1 | soldier | 996 | 101 | 56 | 1,153 |
| 2 | tank | 1,392 | 134 | 61 | 1,587 |
| | **All** | **3,684** | **366** | **188** | **4,238** |

`soldier` and `tank` here are MV class names. They are not the same label ids as KIIT-MiTA or military_object_dataset.

---

## Notes for SENTINEL-X training

1. **KIIT-MiTA** is the smallest and most domain-aligned military-asset set (tank, radar, missile, artillery). Good for a first YOLO/TensorRT loop.
2. **military_object_dataset** is the largest military set, but several classes are missing from val/test. Do not report per-class mAP on warship, weapon, or civilian without fixing the split.
3. Combine the military datasets only after remapping class IDs. Those two label spaces are incompatible as-is.
4. **SULAND v2** is a separate surface-landmine task (`butterfly`, `starfish`). Do not merge it into the CNN military labels.
5. **MV** keeps three class names for YOLO training: armoured personnel carrier, soldier, and tank. The small CNN folds the carrier and tank into military_vehicle, and soldier into camouflage_soldier. Air-fighter and bomber are gone. `Data/YOLO/MV/yolo26m_5class.pt` is the old 5-class run. Do not load it for these labels.
