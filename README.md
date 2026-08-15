# Servicios de ayuda al diagnóstico de GBM

Servicios orientados a brindar soporte al diagnóstico y análisis de **Glioblastoma Multiforme (GBM)** mediante modelos de aprendizaje automático.

## Instalación y configuración

### 1. Descargar los modelos

Descargar los modelos entrenados desde el siguiente enlace:

[Descargar modelos desde Google Drive](https://drive.google.com/file/d/1PZUdZNuJczVNK8AeokyhDdIXTYDPgOnj/view?usp=sharing)

### 2. Organizar los modelos y archivos

Una vez descargados los modelos, colocarlos en el directorio `shared_data` siguiendo la siguiente estructura:

```text
shared_data/
├── atlas/
│   ├── sri24_mask.nii.gz
│   └── sri24_t1.nii.gz
├── results/
├── trained_models/
│   ├── 1dhzmigz_best_model_pipe2/
│   │   └── model.pt
│   ├── vtzpbajf_best_model_pipe1/
│   │   └── model.pt
│   ├── contrastive_projection_head_final_new_pipe1_v01_m1.pth
│   ├── contrastive_projection_head_final_new_pipe2_m1_1dhzmigz.pth
│   ├── supervised_classifier_final_pipe1_v01_m1.pth
│   └── supervised_classifier_final_pipe2_m1_1dhzmigz.pth
└── uploads/
```

### 3. Levantar los microservicios

Desde el directorio raíz del proyecto, ejecutar:

```bash
docker compose up -d
```

Para verificar que los servicios se encuentren correctamente levantados:

```bash
docker compose ps
```

Para consultar los logs de los servicios:

```bash
docker compose logs -f
```

## Estructura de directorios

* `shared_data/atlas/`: contiene los archivos del atlas utilizados por los servicios.
* `shared_data/results/`: almacena los resultados generados por los modelos.
* `shared_data/trained_models/`: contiene los modelos entrenados y sus respectivos pesos.
* `shared_data/uploads/`: directorio destinado a los estudios o archivos cargados para su procesamiento.
