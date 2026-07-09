# Dossier perso Williams — traductions FR des logs

**Ne pas pousser sur le repo qllzz** (déjà dans `.gitignore`).

## Usage

1. Lance la sim normalement (logs chinois dans le terminal).
2. Ouvre en parallèle le fichier qui correspond au script :
   - `challenge_task.md` — démarrage, pince, contrôleurs
   - `scene1_task.md` — tests 1 à 5
   - `simulateur.md` — bruit MPC / Drake / MuJoCo

## Avant push équipe

```bash
git status   # log_fr/ ne doit pas apparaître
git add scripts/ src/   # seulement le code CN
```

## Structure équipe (optionnel)

L'équipe peut créer son propre dossier `log/` (notes CN, debug, etc.)  
sans toucher à `log_fr/` (perso).
