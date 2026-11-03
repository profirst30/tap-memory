# tap-memory : ce qu'une machine doit retenir entre deux passages de téléphone

## Le problème

Beaucoup de machines et d'appareils n'ont pas de connexion permanente. On les lit sur place : un technicien approche son téléphone (NFC) ou s'y connecte un moment (BLE), lit quelques variables du microcontrôleur, puis repart. Entre deux passages, la machine continue de tourner, et une anomalie peut apparaître puis disparaître sans témoin.

La question : **pour un même nombre d'octets lus au passage, que faut-il garder pour que le téléphone sache qu'il s'est passé quelque chose ?** La mémoire d'une petite puce NFC dynamique compte quelques centaines d'octets (512 pour 4 Kbit).

Quatre stratégies sont comparées :

1. **Le dernier état** : moyenne et dispersion de chaque capteur sur les 10 dernières secondes (32 octets). C'est ce qu'on lit quand on lit « les variables actuelles ».
2. **Les dernières mesures brutes** : autant de secondes de mesures que le budget le permet, et le téléphone fait l'analyse.
3. **Un résumé tenu par le microcontrôleur, avec une règle simple** : toutes les 10 s, le plus grand écart d'un capteur à son niveau normal. Le résumé garde le maximum de chaque bloc de fenêtres, sur un octet.
4. **Le même résumé, avec un petit auto-encodeur quantifié en INT8** : il apprend comment les capteurs bougent *ensemble* quand tout va bien, et son score monte quand ils ne s'accordent plus.

## La démarche

- **Données** : SKAB (Skoltech Anomaly Benchmark), un banc de pompe instrumenté. Il fournit 8 capteurs à la seconde (deux accéléromètres, courant, pression, deux températures, tension, débit), 34 expériences d'une vingtaine de minutes avec des anomalies étiquetées (vannes, cavitation, déséquilibre…), et un enregistrement sans anomalie.
- **Calibration à l'installation** : pendant ses 5 premières minutes, chaque machine apprend ses propres niveaux. Chaque capteur est ramené à son niveau et à sa dispersion pendant ces minutes, et le seuil d'alarme vaut un facteur fois le plus grand score vu pendant la calibration.
- **Le facteur** : il est choisi sur l'enregistrement sans anomalie seulement, en rejouant une installation toutes les minutes. C'est le facteur dépassé par environ 10 % de ces installations dans les 10 minutes qui suivent.
- **Deux passages par expérience** : le premier 10 s avant le début de l'anomalie (la période qui précède est normale), le second à la fin de l'enregistrement. Dans la plupart des expériences, l'anomalie est terminée avant le second passage.
- **L'auto-encodeur** : il prend 16 entrées (moyenne et dispersion des 8 capteurs sur 10 s). Il est entraîné sur la première moitié de l'enregistrement sans anomalie, exporté en ONNX puis quantifié en INT8.
- **Le résumé** (`src/core.py`, fonction `summary`) : 8 octets d'en-tête, puis un octet par bloc de fenêtres. L'en-tête contient le nombre de fenêtres en alarme, la première et la dernière, et le capteur le plus en cause. Quand la période s'allonge, les blocs s'allongent : le résumé ne dépasse jamais le budget.

## Résultats

Tous les chiffres ci-dessous sont écrits par `python -m src.run`.

<!-- results -->
33 expériences sur 34 ; 33 périodes avec anomalie (durée médiane 9,7 min) et 33 périodes normales.

| ce que le téléphone lit | octets | anomalies vues | dont anomalie finie avant le passage | dont anomalie en cours | fausses alarmes | erreur médiane sur le début |
|---|---:|---:|---:|---:|---:|---:|
| dernier état (moyenne et dispersion des 10 dernières secondes) | 32 | 2 sur 33 | 2 sur 31 | 0 sur 2 | 0 sur 33 | – |
| dernières mesures brutes | 64 | 2 sur 33 | 2 sur 31 | 0 sur 2 | 0 sur 33 | – |
| dernières mesures brutes | 128 | 1 sur 33 | 1 sur 31 | 0 sur 2 | 0 sur 33 | – |
| dernières mesures brutes | 256 | 1 sur 33 | 1 sur 31 | 0 sur 2 | 0 sur 33 | – |
| dernières mesures brutes | 512 | 2 sur 33 | 2 sur 31 | 0 sur 2 | 0 sur 33 | – |
| résumé, règle simple (plus grand écart d'un capteur) | 61 | 18 sur 33 | 17 sur 31 | 1 sur 2 | 2 sur 33 | 50,0 s |
| résumé, règle simple (plus grand écart d'un capteur) | 84 | 18 sur 33 | 17 sur 31 | 1 sur 2 | 2 sur 33 | 50,0 s |
| résumé, auto-encodeur INT8 | 61 | 19 sur 33 | 19 sur 31 | 0 sur 2 | 2 sur 33 | 50,0 s |
| résumé, auto-encodeur INT8 | 84 | 19 sur 33 | 19 sur 31 | 0 sur 2 | 2 sur 33 | 50,0 s |

Seuil de chaque machine : facteur × le plus grand score vu pendant ses 5 minutes de calibration (auto-encodeur ×5,848, règle ×3,211). Les facteurs sont choisis sur 64 installations rejouées dans l'enregistrement sans anomalie, pour qu'environ 10 % d'entre elles donnent une fausse alarme dans les 10 minutes.

Compromis complet, en changeant le facteur (anomalies vues / fausses alarmes, sur les mêmes périodes) :

| facteur | 1 | 1,5 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| résumé, auto-encodeur | 33 / 25 | 31 / 11 | 29 / 7 | 27 / 3 | 24 / 3 | 19 / 2 | 15 / 0 | 13 / 0 | 11 / 0 |
| résumé, règle simple | 31 / 24 | 27 / 9 | 26 / 4 | 18 / 2 | 16 / 1 | 11 / 0 | 9 / 0 | 7 / 0 | 6 / 0 |
| dernier état | 16 / 11 | 9 / 3 | 7 / 1 | 3 / 1 | 2 / 1 | 2 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |

Capteur le plus en cause selon le résumé (auto-encodeur, 512 octets), par série d'expériences :

- other : vibration 1 (5), débit (1), thermocouple (1)
- valve1 : débit (8), vibration 2 (2)
- valve2 : débit (2)

Résolution du résumé quand la période entre deux passages s'allonge (secondes couvertes par un octet) :

| budget | 1 heure | 1 jour | 1 semaine |
|---:|---:|---:|---:|
| 64 octets | 70 s | 1 550 s | 10 800 s |
| 128 octets | 30 s | 720 s | 5 040 s |
| 256 octets | 20 s | 350 s | 2 440 s |
| 512 octets | 10 s | 180 s | 1 200 s |

Sur AMD Ryzen 7 5825U with Radeon Graphics, un thread : une fenêtre de 10 s coûte 12,8 µs en FP32 et 17,3 µs en INT8 (356 paramètres, 320 multiplications-additions ; fichier ONNX de 2 263 octets en FP32, 5 820 en INT8). INT8 et FP32 prennent la même décision sur 100,0 % des fenêtres.
<!-- /results -->

**En résumé** : ce que le microcontrôleur garde compte davantage que le modèle. Le dernier état et les mesures brutes ne montrent presque jamais une anomalie déjà terminée, alors qu'un résumé de moins de 90 octets en montre la majorité. Au point de fonctionnement choisi, l'auto-encodeur et la règle simple font presque jeu égal. Sur l'ensemble du compromis, l'auto-encodeur voit plus d'anomalies pour le même nombre de fausses alarmes : sans doute parce qu'il repère des capteurs qui se désaccordent sans qu'aucun ne sorte franchement de sa plage (hypothèse, non vérifiée capteur par capteur). Pour les expériences de vannes, le capteur désigné est le plus souvent le débit.

## Limites

- **Périodes courtes** : les expériences durent une vingtaine de minutes, donc le budget d'octets ne contraint pas le résumé. La résolution pour des périodes d'une heure à une semaine est calculée, pas testée.
- **Un seul banc, une seule pompe** : une trentaine de périodes avec anomalie, c'est peu, et la proportion d'anomalies vues varie beaucoup selon le facteur choisi (voir la courbe de compromis).
- **Une expérience écartée** : son anomalie commence pendant les 5 minutes de calibration.
- **Pas de matériel testé** : aucune puce NFC, aucun module radio, aucun microcontrôleur. Coût mesuré avec ONNX Runtime sur un processeur de PC. Sur ce processeur, la version INT8 n'est pas plus rapide que la FP32 pour un réseau aussi petit, et son fichier ONNX est même plus gros (les nœuds de quantification pèsent plus que les poids).
- **Fausses alarmes sous-estimées** : les périodes normales ne durent que quelques minutes. Sur une journée entre deux passages, il y aurait davantage d'occasions de fausse alarme.

## Reproduire

```
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate sous Linux
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m src.fetch     # télécharge SKAB (environ 5 Mo)
python -m src.run       # entraîne, rejoue les passages, écrit results/ et docs/
python -m pytest -q
```

La page de démo (`docs/index.html`) lit `docs/results.json` et `docs/series.json`.

## Données

Katser, I. D., Kozitsin, V. O. *Skoltech Anomaly Benchmark (SKAB)*, Kaggle, 2020, https://github.com/waico/SKAB. Licence GPL-3.0. Le dépôt ne contient pas les données : `src/fetch.py` les télécharge.
