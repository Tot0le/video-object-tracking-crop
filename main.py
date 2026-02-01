import cv2
import numpy as np
import os
import glob

# --- CONFIGURATION PERFORMANCE ---
OUTPUT_VIDEO = "celeste_fast.mp4"
SMOOTHING = 0.15
MATCH_THRESHOLD = 0.70

# FACTEUR DE VITESSE :
# 0.5 = 2x plus rapide (Analyse sur image 540p)
# 0.25 = 4x plus rapide (Analyse sur image 270p) -> Recommandé si ça lag
PROCESS_SCALE = 0.25 

def loadRefImages(folderPath):
    images = []
    cleanPath = folderPath.strip('"').strip("'")
    absPath = os.path.abspath(cleanPath)

    print(f"--- Chargement des sprites ---")
    if os.path.exists(absPath):
        types = ['*.png', '*.jpg', '*.jpeg']
        files = []
        for t in types:
            pattern = os.path.join(absPath, '**', t)
            foundFiles = glob.glob(pattern, recursive=True)
            files.extend(foundFiles)
        files.sort()
        
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                images.append(img)
                print(f" -> Chargé : {os.path.basename(f)}")
    else:
        print(f"❌ Dossier introuvable : {absPath}")
    return images

def processVideoFast(inputPath, spritesPath, outputPath):
    # 1. Vérifications
    cleanVideoPath = inputPath.strip('"').strip("'")
    absVideoPath = os.path.abspath(cleanVideoPath)
    
    if not os.path.exists(absVideoPath):
        print(f"❌ ERREUR : Vidéo introuvable.")
        return

    cap = cv2.VideoCapture(absVideoPath)
    if not cap.isOpened():
        print("❌ ERREUR : Impossible d'ouvrir la vidéo.")
        return

    srcWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    srcHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✅ Vidéo chargée : {srcWidth}x{srcHeight} | {totalFrames} frames.")
    print(f"🚀 Mode Turbo activé (Echelle de calcul : {PROCESS_SCALE})")

    finalOutWidth = 1080
    finalOutHeight = 1920
    cropHeight = srcHeight
    cropWidth = int(srcHeight * 9 / 16)

    dirName = os.path.dirname(absVideoPath)
    finalOutPath = os.path.join(dirName, outputPath)
    
    # Correction VS Code (ignorer l'erreur visuelle)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(finalOutPath, fourcc, fps, (finalOutWidth, finalOutHeight))

    # --- PREPARATION DES SPRITES ---
    ret, frame = cap.read()
    if not ret: return

    # Liste HD (pour sauvegarde) et Liste LOW RES (pour recherche rapide)
    templates_hd = loadRefImages(spritesPath)
    templates_small = []

    # On redimensionne tous les sprites chargés pour qu'ils matchent la petite résolution
    for t in templates_hd:
        small_t = cv2.resize(t, (0,0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        templates_small.append(small_t)

    camX = srcWidth / 2.0

    # Si pas de sprites, sélection manuelle
    if len(templates_hd) == 0:
        print("\n⚠️ Aucun sprite. Sélectionne Madeline sur l'image (HD).")
        bbox = cv2.selectROI("Selection Initiale", frame, showCrosshair=True)
        cv2.destroyWindow("Selection Initiale")
        # Si sélection valide
        if bbox[2] > 0:
            tX, tY, tW, tH = [int(v) for v in bbox]
            hd_temp = frame[tY:tY+tH, tX:tX+tW]
            
            # On ajoute aux deux listes
            templates_hd.append(hd_temp)
            small_temp = cv2.resize(hd_temp, (0,0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
            templates_small.append(small_temp)
            
            camX = float(tX + tW / 2)
    else:
        print(f"\n✅ {len(templates_hd)} sprites chargés.")

    print("\n--- TRAITEMENT RAPIDE ---")
    print("Appuie sur 'q' pour quitter, 's' pour recalibrer, 'a' pour ajouter.")

    frameCount = 0
    
    while True:
        if frameCount > 0:
            ret, frame = cap.read()
            if not ret: break

        # --- 1. DOWNSCALING (Le secret de la vitesse) ---
        # On crée une image minuscule juste pour l'analyse
        small_frame = cv2.resize(frame, (0,0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)

        bestScore = -1.0
        bestPoint = None
        bestW = 0
        
        # --- 2. DETECTION SUR PETITE IMAGE ---
        if len(templates_small) > 0:
            for temp in templates_small:
                res = cv2.matchTemplate(small_frame, temp, cv2.TM_CCOEFF_NORMED)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
                
                if maxVal > bestScore:
                    bestScore = maxVal
                    bestPoint = maxLoc
                    bestW = temp.shape[1]

        # --- 3. CONVERSION DES COORDONNEES ---
        targetX = camX # Par défaut on bouge pas
        isLost = False
        
        if bestScore >= MATCH_THRESHOLD and bestPoint is not None:
            # On a trouvé Madeline en petit, on remet à l'échelle x4
            smallCenterX = bestPoint[0] + bestW // 2
            targetX = float(smallCenterX / PROCESS_SCALE)
        else:
            isLost = True

        # --- 4. LISSAGE & DECOUPE HD ---
        camX = camX + (targetX - camX) * SMOOTHING

        x1 = int(camX - cropWidth / 2)
        if x1 < 0: x1 = 0
        elif x1 + cropWidth > srcWidth: x1 = srcWidth - cropWidth
        
        # On découpe TOUJOURS dans l'image HD originale pour la qualité
        crop = frame[0:srcHeight, x1:x1+cropWidth]
        finalFrame = cv2.resize(crop, (finalOutWidth, finalOutHeight), interpolation=cv2.INTER_NEAREST)
        out.write(finalFrame)

        # --- 5. AFFICHAGE LEGER ---
        # On affiche seulement 1 image sur 2 pour laisser le PC respirer
        if frameCount % 2 == 0:
            # On dessine sur l'image finale
            debugFrame = finalFrame.copy()
            color = (0, 255, 0) if not isLost else (0, 0, 255)
            # Petit rond vert/rouge en haut à gauche
            cv2.circle(debugFrame, (50, 50), 20, color, -1) 
            
            if isLost:
                cv2.putText(debugFrame, "PERDU", (80, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Redimensionner pour l'écran
            previewH = 600
            previewW = int(previewH * 9 / 16)
            display = cv2.resize(debugFrame, (previewW, previewH))
            
            cv2.imshow("Apercu Rapide", display)
        
        # C'est cette commande qui empêche la fenêtre de figer
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Recalibrage sur image HD
            newBox = cv2.selectROI("Recalibrer Camera", frame, showCrosshair=True)
            cv2.destroyWindow("Recalibrer Camera")
            if newBox[2] > 0:
                camX = float(newBox[0] + newBox[2] / 2)

        elif key == ord('a'):
            # Ajout sur image HD
            newBox = cv2.selectROI("Ajouter Ref", frame, showCrosshair=True)
            cv2.destroyWindow("Ajouter Ref")
            if newBox[2] > 0:
                tX, tY, tW, tH = [int(v) for v in newBox]
                hd_temp = frame[tY:tY+tH, tX:tX+tW]
                
                # Ajout HD
                templates_hd.append(hd_temp)
                # Creation version petite + Ajout
                small_temp = cv2.resize(hd_temp, (0,0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
                templates_small.append(small_temp)
                
                camX = float(tX + tW / 2)
                print("Pose ajoutée !")

        frameCount += 1
        # Log propre
        if frameCount % 30 == 0:
             print(f"Progression : {frameCount}/{totalFrames} frames", end='\r')

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nTerminé !")

if __name__ == "__main__":
    print("=== AUTO-CROPPER CELESTE (MODE TURBO) ===")
    vInput = input("📂 Glisse ta vidéo ici : ")
    sInput = input("📂 Dossier sprites (Entrée pour auto) : ")
    
    if sInput.strip() == "":
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        sInput = os.path.join(scriptDir, "sprites")
    
    processVideoFast(vInput, sInput, OUTPUT_VIDEO)