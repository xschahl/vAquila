# Contexte Projet : vAquila 🦅

## Mission Produit

Tu es l'assistant IA de **vAquila**, un orchestrateur open-source de LLMs qui vise :

- l'expérience développeur d'`Ollama` (simple, rapide),
- avec la robustesse production de **vLLM + Docker**.

La commande CLI principale est `vaq`.

---

## Stack & Contraintes Techniques

- **Python**: 3.10+
- **CLI**: `Typer`
- **Affichage**: `rich`
- **Orchestration conteneurs**: SDK officiel `docker` Python
- **GPU/NVML**: `pynvml`
- **Runtime vLLM**: conteneur `vllm/vllm-openai`

### Règle d'or Docker

Toujours privilégier l'API Python Docker (pas de shell `subprocess`/`os.system`) quand l'API couvre le besoin.

---

## État Actuel du Produit (important)

### Commandes actives

- `vaq run <model_id>`
- `vaq ps`
- `vaq stop <model_id> [--purge-cache]`
- `vaq list`
- `vaq rm <model_id>`
- `vaq doctor`
- `vaq infer`

### Rebalance

- Le workflow **rebalance n'est plus exposé en CLI**.
- Ne pas réintroduire de comportement de rééquilibrage automatique dans `vaq run`.
- En cas de VRAM insuffisante, prioriser un message clair demandant de libérer la VRAM (`vaq ps` / `vaq stop`).

---

## Invariants Fonctionnels (à respecter absolument)

1. **GPU safety first**
   - Lire l'état VRAM via NVML avant tout lancement.
   - Réserver un buffer de sécurité OS/processus.
   - Ne jamais hardcoder un ratio fixe unique de VRAM.

2. **Erreurs utilisateur propres**
   - Jamais de traceback brut pour l'utilisateur final.
   - Lever `VaquilaError` puis afficher via `rich`/`typer`.
   - Messages explicites et actionnables.

3. **Persistance des modèles HF**
   - Monter **obligatoirement** le cache HF host vers `/root/.cache/huggingface` dans le conteneur.
   - Objectif: éviter les retéléchargements.

4. **Typage strict et docstrings**
   - `Type hints` partout.
   - Docstrings concises, orientées maintenance.

5. **Langue UX**
   - Les messages utilisateur, aides CLI et docstrings visibles sont en **anglais professionnel**.

---

## Architecture Pratique (fichiers clés)

- `src/vaquila/cli.py`: options/commandes Typer
- `src/vaquila/commands/run.py`: orchestration complète de `vaq run`
- `src/vaquila/helpers/runtime.py`: résolution options runtime + heuristiques ratio
- `src/vaquila/helpers/context.py`: stratégie dépassement contexte modèle
- `src/vaquila/helpers/startup.py`: parsing logs startup vLLM
- `src/vaquila/docker_service.py`: création/arrêt/listing conteneurs
- `src/vaquila/gpu.py`: snapshots NVML + calcul ratio VRAM

---

## Détails Critiques du Flux `vaq run`

### 1) Résolution runtime

- Résoudre `max_num_seqs`, `max_model_len`, parsers, thinking mode.
- Gérer `quantization` (`auto`, explicite, `none`, etc.).
- Gérer `kv_cache_dtype` (`fp16` ou `fp8`).

### 2) Validation contexte modèle

- Vérifier la limite contexte depuis config HF (cache local puis Hub).
- Si dépassement: proposer clamp ou override long-context.

### 3) Estimation ratio initial

- Calculer ratio requis depuis paramètres runtime.
- Compléter par estimation analytique profil modèle quand possible (weights + KV cache + overhead).

### 4) Pré-check VRAM

- Calculer ratio max disponible depuis snapshot NVML + buffer sécurité.
- Refuser le lancement si configuration impossible avant startup vLLM.

### 5) Startup retry orienté données vLLM

- En cas d'échec KV cache, parser les logs/erreurs (`needed GiB` vs `available KV cache memory`).
- Proposer un ratio suivant intelligent (pas uniquement incréments fixes).

### 6) Post-start tuning

- Lire la concurrence observée (`Maximum concurrency for ... tokens per request`).
- Ajuster le ratio dans une fenêtre bornée pour éviter sur-allocation/sous-allocation.

### 7) Persistance

- Sauvegarder ratio stable par profil de lancement (model + seqs + context + quant + kv dtype) pour accélérer les runs suivants.

---

## Règles d'Implémentation pour l'Agent IA

- Faire des changements **ciblés et minimaux**.
- Ne pas refactorer hors-scope.
- Préserver les labels Docker `com.vaquila.*` existants.
- Si ajout d'option CLI: propager jusqu'au `docker_service.run_model_container`.
- Si un comportement vLLM change: mettre à jour README + help CLI.
- Garder la compatibilité Windows (chemins cache host lisibles par daemon Docker).

---

## Validation attendue après modification

1. Vérifier erreurs statiques/fichiers modifiés.
2. Vérifier `docker compose run --rm vaq run --help`.
3. Si changement `run`: test lancement réel d'au moins un modèle léger.
4. Vérifier `vaq ps` / `vaq stop` sur le conteneur lancé.

---

## Anti-patterns à éviter

- Introduire des appels shell quand l'API Docker suffit.
- Réintroduire le rebalance CLI/auto dans le flux principal.
- Coder des ratios VRAM magiques non justifiés.
- Retourner des erreurs vagues sans action recommandée.
- Mélanger des messages FR/EN côté utilisateur.

---

## Résumé exécutable pour l'agent

Quand tu touches `vaq run`, pense systématiquement: **NVML safety -> estimation réaliste -> startup guidé par erreurs vLLM -> tuning borné -> message utilisateur clair -> persistance des paramètres utiles**.

---

## Playbooks Opérationnels (à appliquer sans hésiter)

### Playbook A — `vaq run` échoue avant startup vLLM

1. Vérifier que l'échec vient du pré-check VRAM (ratio max dispo < ratio requis).
2. Ne pas tenter de "forcer" le run.
3. Retourner un message actionnable:
   - réduire `--max-num-seqs` et/ou `--max-model-len`,
   - ou libérer la VRAM avec `vaq ps` puis `vaq stop <model_id>`.
4. Garder l'erreur claire via `VaquilaError` (pas de traceback).

### Playbook B — Erreur KV cache au startup vLLM

1. Détecter si l'erreur est retryable (`available KV cache memory`, `No available memory for cache blocks`, etc.).
2. Parser les métriques vLLM (`needed GiB` vs `available GiB`) si présentes.
3. Proposer un ratio suivant calculé (avec marge), pas un incrément arbitraire fixe.
4. Relancer jusqu'à trouver la plus petite config stable dans les bornes disponibles.
5. Journaliser clairement chaque tentative côté CLI.

### Playbook C — Sur-allocation après startup

1. Lire la concurrence observée depuis les logs (`Maximum concurrency for ...`).
2. Comparer à la concurrence demandée (`max_num_seqs`) avec une petite marge.
3. Si trop élevée: diminuer progressivement le ratio.
4. Si trop basse: augmenter prudemment le ratio.
5. Arrêter la boucle quand la fenêtre stable/fail devient trop étroite.

### Playbook D — Persistance des bons réglages

1. Construire une clé de profil stable: `model + seqs + context + quant + kv dtype`.
2. Recharger un ratio stable connu au prochain lancement si compatible avec VRAM dispo.
3. Sauvegarder le ratio final stable en fin de run.
4. Ne jamais persister un ratio qui a échoué.

### Playbook E — Ajout/modif d'option CLI `run`

Checklist obligatoire:

1. Ajouter l'option dans `src/vaquila/cli.py`.
2. Propager la valeur jusqu'à `cmd_run`.
3. Propager jusqu'à `docker_service.run_model_container`.
4. Ajouter flags vLLM (`command`) et labels `com.vaquila.*` si pertinent.
5. Mettre à jour README + `--help` avec wording pro en anglais.

### Playbook F — Compatibilité Windows / Docker Desktop

1. Vérifier que `VAQ_HF_CACHE_HOST_PATH` est lisible par le daemon Docker.
2. Éviter de casser les chemins absolus Windows (`C:/...`).
3. Ne pas convertir aveuglément en chemins Linux côté host.
4. Conserver le mount HF vers `/root/.cache/huggingface` dans le conteneur.

### Playbook G — Indisponibilité Docker ou NVML

1. Capturer `DockerException` / `NVMLError`.
2. Lever `VaquilaError` avec action utilisateur explicite (démarrer Docker, vérifier drivers/GPU).
3. Afficher proprement via `rich`/`typer`.
4. Ne jamais exposer une stacktrace brute à l'utilisateur CLI.

### Playbook H — Quand ne rien faire

- Ne pas réintroduire `rebalance` dans la CLI.
- Ne pas ajouter de logique hors-scope si la demande concerne uniquement messages/docs.
- Ne pas refactorer massivement un module pour un petit changement.
