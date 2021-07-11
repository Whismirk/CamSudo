# -*- coding: utf-8 -*-
import cv2
import numpy
import operator
import random
from tensorflow import keras
from tensorflow.python.keras.models import Model
import solver

#MODEL = keras.models.load_model("models/model_mnist.h5")
MODEL = keras.models.load_model("models/model_typed.h5")

# Police utilisé pour afficher les chiffres sur la grille.
FONT = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

capture = cv2.VideoCapture(0)
# Dimensions en pixels de la fenêtre affichant la capture vidéo.
CAPTURE_DISPLAY_X = 960
CAPTURE_DISPLAY_Y = 540

# Dimensions attendues de la grille, en unités arbitraires.
CASE_MARGIN = 4
CASE_SIZE = 28 + 2 * CASE_MARGIN
GRID_LENGTH = 9 * CASE_SIZE

# Dimensions utilisées pour le flou gaussien.
GAUSSIAN_KERNEL_SIZE = (5, 5)

# Seuil (encore une fois arbitraire) à partir duquel on considère que la case est (ou n'est pas) vide, pour le CNN.
CASE_EMPTY_TRESHOLD = 20000

# Aire minimum pour qu'une forme soit considérée comme étant potentiellement la grille.
AREA_MIN = 25000

# Pour éviter de re-résoudre inutilement la même grille.
solved = False
# Lorsque la grille n'est plus détectée, combien d'itérations attendre avant de réinitialiser le statut de résolution.
# Permet de laisser un court laps de temps (parce qu'on a trop bougé la feuille, par exemple...) avant d'avoir à tout recommencer,
# tout en permettant de passer une autre grille sans devoir relancer le programme.
# En gros, si on détecte à nouveau une grille avant que le temps soit écoulé, on considère qu'il s'agit de l'ancienne.
RESET_TICKS = 100
reset_timer = 0

# Boucle principale, chaque itération représentant le traitement d'une frame de la capture vidéo de la caméra.
while True:
    # frame : l'image vidéo actuelle.
    # retval : booléen, a-t-on bien réussi à obtenir une frame ? 
    retval, frame = capture.read()

    # On transforme la frame en niveau de gris, pour effecturer le seuillage par la suite.
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # On adoucit l'image en réduisant les détails
    grayscaled_frame = cv2.GaussianBlur(grayscaled_frame, GAUSSIAN_KERNEL_SIZE, 0)

    # On effectue le seuillage pour transformer l'image en noir et blanc.
    # Tous les pixels possédant un niveau de gris/noir supérieur à un certain seuil deviendront complètement noir(255), les autres blancs (0).
    # Seuillage adaptatif ici pour éviter les problèmes dûs à la luminosité si on bouge trop la grille (par exemple, à cause de mes mains qui tremblent...).
    threshold_frame = cv2.adaptiveThreshold(grayscaled_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    
    # On demande à OpenCv de détecter des contours au sein de l'image.
    contours, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # On initialise les contours que l'on cherche (celles de la grille).
    contours_grid = None

    # On part du principe que le plus grand carré détecté dans l'image sera généralement la grille.
    area_max = AREA_MIN

    # On itère sur chaque contours trouvé dans l'image pour trouver la grille.
    for contour in contours:
        # On filtre les plus petites aires pour ne récupérer que les plus importantes formes de l'image.
        area = cv2.contourArea(contour)
        if area > area_max:

            # On calcule le périmètre de la forme trouvée.
            perimeter = cv2.arcLength(contour, True)
            # On cherche un polygone fermé à partir de cette forme.
            # On demande à OpenCv de trouver un polygone dont le périmètre correspond à celui de la forme, à 1% près.
            polygon = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

            # Si le polygone possède 4 côtés, on le retient, et on réitère.
            if len(polygon) == 4:
                # Pour le moment, ce polygone correspond le mieux à la grille (que l'on cherche).
                contours_grid = polygon
                # On met à jour la plus grande aire.
                area_max = area

    # Si aucune grille détectée, on affiche simplement la frame.
    if contours_grid is None:
        if solved:
            # On vérifie le nombre d'itérations passées. Si on dépasse le seuil, on réinitialise l'état de solution, sinon on incrémente.
            if reset_timer >= RESET_TICKS :
                solved = False
                print("État de résolution réinitialisé. Vous pouvez passer une nouvelle grille !")
            else:
                reset_timer += 1
                reset_time = RESET_TICKS - reset_timer
                print("Grille perdue. Réinitialisation dans {}.".format(reset_time))

        frame = cv2.resize(frame, (CAPTURE_DISPLAY_X, CAPTURE_DISPLAY_Y))
        cv2.imshow("Capture", frame)

    else:
        # On remet le timer à zéro, puisque qu'on a détecté une grille.
        reset_timer = 0

        # On affiche les bordures de la grille en overlay sur l'image.
        cv2.drawContours(frame, [contours_grid], 0, (0, 255, 0), 2)

        # points_from : On récupère les 4 points aux 'angles' de la grille.
        points_from = numpy.vstack(contours_grid).squeeze()
        points_from = sorted(points_from, key=operator.itemgetter(1))
        # points_target : L'emplacement de destination des points une fois la transformation de perspective effectuée.
        points_target = numpy.float32([[0, 0], [GRID_LENGTH, 0], [0, GRID_LENGTH], [GRID_LENGTH, GRID_LENGTH]])
        # On s'assure que les points sont dans l'ordre (même après le sort), nécessaire pour la transformation de perspective.
        if points_from[0][0] < points_from[1][0]:
            if points_from[3][0] < points_from[2][0]:
                points_from = numpy.float32([points_from[0], points_from[1], points_from[3], points_from[2]])
            else:
                points_from = numpy.float32([points_from[0], points_from[1], points_from[2], points_from[3]])
        else:
            if points_from[3][0] < points_from[2][0]:
                points_from = numpy.float32([points_from[1], points_from[0], points_from[3], points_from[2]])
            else:
                points_from = numpy.float32([points_from[1], points_from[0], points_from[2], points_from[3]])
        
        # On effectue une transformation de perspective pour avoir une image centrée sur la grille, grâce aux points récupérés précedemment.
        matrix_transform = cv2.getPerspectiveTransform(points_from, points_target)
        grid_image = cv2.warpPerspective(frame, matrix_transform, (GRID_LENGTH, GRID_LENGTH))
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
        grid_image = cv2.adaptiveThreshold(grid_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)
        cv2.imshow("Grid", grid_image)
        
        # C'est parti !
        # On détermine le contenu de la grille et on résout le sudoku.
        if not solved:
            grid_origin = []
            # On itère sur 9x9.
            for y in range(9):
                line = ""
                for x in range(9):
                    # On détermine les coordonnées de la case courante.
                    x_min = x * CASE_SIZE + CASE_MARGIN
                    x_max = (x + 1) * CASE_SIZE - CASE_MARGIN
                    y_min = y * CASE_SIZE + CASE_MARGIN
                    y_max = (y + 1) * CASE_SIZE - CASE_MARGIN
                    # On adapte, à l'aide de numpy, la case de l'image transformée à la forme d'entrée demandée par le modèle du CNN.
                    cnn_input = grid_image[y_min:y_max, x_min:x_max].reshape(1, 28, 28, 1)
                    
                    # On vérifie que la case n'est pas vide.
                    if cnn_input.sum() > CASE_EMPTY_TRESHOLD:
                        # On utilise le modèle pour déterminer le chiffre dans la case.
                        predictions = MODEL.predict(cnn_input)
                        probability_max = numpy.amax(predictions)
                        best_candidate = numpy.argmax(predictions, axis=-1)[0]
                        #if (best_candidate == 6 or best_candidate == 8):
                            #best_candidate = random.choice([6, 8])
                        line += "{:d}".format(best_candidate)
                    # Si la case est vide, on le note par un zéro.
                    else:
                        line += "{:d}".format(0)
                grid_origin.append(line)
            print("Grille d'origine : {}".format(grid_origin))
            grid_result = solver.solve(grid_origin)

        # On a réussi, youhou !
        # Maintenant, on affiche les chiffres de la solution en les superposant sur la grille.
        if grid_result is not None:
            solved = True
            print("Grille résolue : {}".format(grid_result))
            matrix_zeros = numpy.zeros(shape=(GRID_LENGTH, GRID_LENGTH, 3), dtype=numpy.float32)
            # On parcourt la grille non résolue pour récupérer les cases vides (donc les 0), afin d'y afficher les chiffres correspondants. 
            for y in range(len(grid_origin)):
                for x in range(len(grid_origin[y])):
                    if grid_origin[y][x] == "0":
                        cv2.putText(matrix_zeros, "{:d}".format(grid_result[y][x]), ((x) * CASE_SIZE + CASE_MARGIN + 3, (y + 1) * CASE_SIZE - CASE_MARGIN - 3), FONT, 0.9, (0, 0, 255), 1)
            # On effectue l'inverse de la transformation effectuée plus tôt pour 'orienter' la superposition.
            matrix_transform_reverse = cv2.getPerspectiveTransform(points_target, points_from)
            frame_height, frame_width, frame_channels = frame.shape
            overlay = cv2.warpPerspective(matrix_zeros, matrix_transform_reverse, (frame_width, frame_height))
            overlay_grayscale = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            # On génère un masque à partir de l'overlay pour ne garder que les chiffres.
            ret, mask = cv2.threshold(overlay_grayscale, 10, 255, cv2.THRESH_BINARY)
            mask = mask.astype('uint8')
            # On applique le masque pour obtenir l'image finale à superposer.
            overlay = cv2.bitwise_and(overlay, overlay, mask=mask).astype('uint8')
            # On génère et applique l'inverse du masque à l'image principale.
            mask_reverse = cv2.bitwise_not(mask)
            image = cv2.bitwise_and(frame, frame, mask=mask_reverse)
            # On combine les deux images pour obtenir le résultat attendu.
            image_combined = cv2.add(image, overlay)
            image_combined = cv2.resize(image, (CAPTURE_DISPLAY_X, CAPTURE_DISPLAY_Y))
            cv2.imshow("Capture", image_combined)

        else:
            frame = cv2.resize(frame, (CAPTURE_DISPLAY_X, CAPTURE_DISPLAY_Y))
            cv2.imshow("Capture", frame)

    # On presse Q pour quitter, et sortir de la boucle.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# C'est fini.
# Hop, on casse tout !
capture.release()
cv2.destroyAllWindows()
