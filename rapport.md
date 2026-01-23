## Modifications apportées

### `diffusion/losses.py`

- Ajout des fonctions pour convertir entre vecteurs 6D et quaternions  
- Ajout des fonctions pour les opérations entre quaternions (inverse, multiplication)  
- Ajout de la distance géodésique  

### `diffusion/gaussian_diffusion.py`

- intégration de la loss géodésique  
- intégration de pondération en fonction du stade de débruitage  

### `utils/parser_util.py`

- ajout du paramètre lambda_geo  

### `eval/eval_humanml.py`

- ajout d'une métrique dédiée à la distance géodésique (degrés)  

## Bilan

Nous avons commencé par vérifier la reproductibilité de MDM, nous avons téléchargé et configuré les bases comme indiqué dans le github.

Ensuite, nous nous sommes intéressés à l'évaluation, après un peu de debugging nous avons réussi à reproduire les résultats présentés dans le papier.

Enfin, avant de commencer à apporter nos modifications, nous avons lancé un entraînement pour s'assurer que tout fonctionne.

Nous avons ensuite regardé les dossiers/fichiers les plus importants pour comprendre comment le modèle fonctionnait. Nous avions ensuite une semaine pour apporter nos modifications et les évaluer. 

Après 3 jours d'ajout/debuggage, nous avons réussi à lancer un entraînement sans erreurs. Mais nous nous sommes rendu compte qu'un entraînement complet prend 2–3 jours, ce qui nous a conduits à fortement réduire le nombre de steps, de 600k à 50k.

Nous avons lancé 3 entraînements :

- un sans modifications pour pouvoir comparer à 50k  
- un avec notre nouvelle loss géodésique (approche simple, lambda_geo=0.1)  
- un avec notre loss + pondération en fonction du stade de débruitage  

Comme le nombre de steps est bien trop petit pour avoir des résultats conclusifs, nous avons quand même ces observations :

En se basant sur le FID (étant une métrique qui mesure la similitude à la distribution réelle), on remarque que notre loss augmente le score (de 1.1, à 1.4/1.7). Nous avons donc eu l'idée d'ajouter une métrique orientée géodésique, et nous avons obtenu une amélioration de 50 degrés en moyenne à 46. Nous pensons que ces résultats seront meilleurs si nous lançons un entraînement à 600k steps.

## Pistes d'améliorations

Les résultats que nous avons obtenus ne nous permettent pas de savoir si  cette nouvelle loss apporte un quelconque bénéfice, cependant, nous pensons que cette loss peut être pertinente au modèle avec un peu plus de travail, comme:

- exclure la loss rot_mse pour les rotations (au lieu de les ajouter)  
- faire la diffusion sur SO(3)  
- essayer ces modifications sur le modèle de 1000 diffusion_steps (nous avons travaillé avec celui de 50 diffusion_steps pour plus de rapidité)  
- entraînement sur 600k steps pour avoir des résultats concrets  