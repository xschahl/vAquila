# Contexte du Projet : vAquila 🦅

## Identité du Projet

Tu es l'assistant IA de "vAquila", un outil d'orchestration de modèles d'IA open-source.
Le but de vAquila est de fournir une interface simple (CLI + Web UI) pour déployer des LLMs en production en utilisant **vLLM** et **Docker**. L'objectif est d'offrir l'expérience développeur de l'outil `Ollama` tout en garantissant des performances de production via vLLM.
La commande principale de la CLI est `vaq`.

## Architecture & Stack Technique

- **Langage** : Python 3.10+
- **CLI** : Utilisation de `Typer` pour créer des commandes élégantes, typées et faciles à maintenir.
- **Moteur d'orchestration** : Le SDK Python `docker` (`import docker`). Règle stricte : PAS d'appels à des scripts shell via `subprocess` ou `os.system` si l'API Docker officielle en Python peut le faire proprement.
- **Gestion GPU** : Utilisation de `pynvml` pour interagir avec les drivers NVIDIA, lire la VRAM totale/libre, et calculer dynamiquement le paramètre `--gpu-memory-utilization` de vLLM avant chaque lancement.

## Règles de Développement Strictes

1. **Sécurité GPU d'abord** : Avant de lancer un conteneur Docker vLLM, le code doit TOUJOURS vérifier la VRAM disponible via NVML et réserver un "buffer" de sécurité (ex: 1 à 2 Go) pour l'OS et les autres processus. Ne jamais hardcoder le ratio de VRAM.
2. **Gestion des erreurs gracieuse** : Si le démon Docker n'est pas lancé, ou si les drivers NVIDIA ne sont pas détectés, l'outil doit attraper l'exception et renvoyer une erreur explicite, colorée et bien formatée pour l'utilisateur (via `rich` ou `typer`), pas une stacktrace Python brute.
3. **Persistance des modèles** : Les conteneurs vLLM générés doivent IMPÉRATIVEMENT monter le dossier local `~/.cache/huggingface` vers `/root/.cache/huggingface` dans le conteneur pour éviter de retélécharger les poids des modèles à chaque redémarrage ou changement de conteneur.
4. **Typage et Documentation** : Utilise le typage strict de Python (`Type hints`) pour toutes les fonctions et rédige des docstrings concises, particulièrement pour les fonctions liées à la gestion hardware et aux appels Docker.

## Commandes Cibles à implémenter (MVP)

- `vaq run <model_id>` : Détecte le GPU, calcule la VRAM, crée le conteneur vLLM avec les bons volumes/ports et le lance en tâche de fond.
- `vaq ps` : Liste les conteneurs vAquila actifs, leur port exposé, et leur consommation VRAM actuelle.
- `vaq stop <model_id>` : Stoppe et supprime le conteneur proprement pour libérer instantanément la ressource GPU.
