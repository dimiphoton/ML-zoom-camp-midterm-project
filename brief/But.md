### But

Le contrat Multirisque Agricole, géré par Pacifica, est souscrit par les agriculteurs pour sécuriser leur exploitation. Il couvre l’activité professionnelle, les dommages aux bâtiments d’exploitation, le matériel stocké, ainsi que la protection financière et juridique. Ce contrat garantit à l’assuré une couverture efficace et durable, assurant ainsi la continuité de son activité en cas de sinistre, tant sur le plan matériel que financier.

Actuellement, le risque d’incendie constitue une part majeure de la charge sinistre du contrat Multirisque Agricole, ce qui en fait un enjeu clé à modéliser avec précision.

L’objectif est d’identifier le meilleur modèle pour prédire la prime pure incendie, en utilisant :

1. Un modèle pour la Fréquence,
2. Un modèle pour le Coût moyen.

**La variable cible finale, la charge, est obtenue en multipliant la fréquence, le coût moyen, et le nombre d’années depuis la souscription du contrat (la variable “ANNEE_ASSURANCE”).**

> **Note** : Un notebook d’exemple est disponible parmi les fichiers à télécharger.

------

### Description des données

Un fichier supplémentaire est mis à disposition regroupant toutes les variables disponibles, accompagnées de leur description. Ce fichier inclut :

- **Les variables cibles** : `FREQ`, `CM`, et `CHARGE`
- **Données géographiques** : département, données météorologiques, etc.
- **Données spécifiques au contrat**, notamment :
  - L’activité de l’assuré (cultivateur, polyculteur, etc.)
  - Les indicateurs de souscription des garanties
  - Le nombre de bâtiments, de salariés, et de sinistres déclarés lors de la souscription
  - **Données de surface** : surfaces des bâtiments (élevage, exploitation, etc.), anonymisées en `surface1`, `surface2`, etc., pour garantir la confidentialité
  - **Données de capitaux** : capitaux assurés pour différentes options (vol, serres, etc.), anonymisés en `capital1`, `capital2`, etc.
  - **Données liées à la prévention** : présence d’équipements (extincteurs, structure en bois, etc.), anonymisées en `prev1`, `prev2`, etc.

------

### Description du benchmark

**Objectif du challenge**
L’objectif de ce challenge est de comparer les performances des modèles développés dans le cadre de cette compétition avec celles d’un modèle de référence basé sur des **GLM (Generalized Linear Models)** classiques.

**Structure du benchmark**
Le benchmark repose sur deux modèles GLM distincts :

1. **Fréquence des sinistres** :
   - Distribution : *Loi de Poisson*
   - Fonction de lien : *Log*
2. **Coût moyen d’un sinistre** :
   - Distribution : *Tweedie*
   - Fonction de lien : *Log*

**Évaluation**
L’évaluation des modèles repose sur une métrique unique : **RMSE (Root Mean Square Error)**, définie par la formule suivante :

RMSE=1n∑i=1n(yi−y^i)2RMSE=√*n*1∑*i*=1*n*(*y**i*−*y*^*i*)2

où :

- **yᵢ** représente la valeur réelle,
- **ŷᵢ** représente la valeur prédite,
- **n** est le nombre d’observations.

L’idée reste d’évaluer dans quelle mesure les approches proposées permettent de dépasser les performances des modèles standards tout en prenant en compte :

- La précision des prédictions
- Les aspects d’interprétabilité et d’efficacité
- Les contraintes métier associées.