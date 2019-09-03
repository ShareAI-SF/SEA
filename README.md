# README pour tester Flask  
Le service web permet d'uploader une image puis de détecter les déchets qui sont dessus. Le résultat de la prédiction est enregistré sous image dans "SEA/FlaskSurfrider/static/img/" et sous forme de json dans "SEA/FlaskSurfrider/static/json/"  

Prérequis :  
1) Avoir git bash  
2) Avoir python  
3) Avoir ces librairies :  
	flask  
	werkzeug  
	numpy  
	os  
	tensorflow 1.14.0  
	matplotlib  
	pillow  
	pandas  
	json  
	opencv-python  

Etapes pour tester flask :  
1) Ouvrir git bash  
2) Clonez le repository  
	> git clone https://github.com/ShareAI-SF/SEA.git  
3) Déplacez vous dans le bon répertoire  
	> cd SEA/FlaskSurfrider  
4) Lancez le serveur  
	> python flasksurfrider.py  
5) 	Accédez à http://127.0.0.1:5000/ dans un navigateur  
	Cliquez sur "Choisir un fichier", il y a déjà trois images dans "SEA/FlaskSurfrider/static/img/" pour tester   
	Après avoir choisi l'image, cliquez sur "Valider"  
	Après 1 ou 2 min l'image prédite s'affichera et sera enregistrée dans "SEA/FlaskSurfrider/static/img/", le résultat en json sera aussi enregistré dans "SEA/FlaskSurfrider/static/json/"  
	Vous pouvez tester une autre image en cliquant sur "Essayer une autre image"  