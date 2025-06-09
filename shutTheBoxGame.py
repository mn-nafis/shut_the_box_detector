import cv2
import threading
import time
from ultralytics import YOLO

openLabels = {
    "oneOpened" : 0,
    "twoOpened" : 1,
    "threeOpened" : 2,
    "fourOpened" : 3,
    "fiveOpened" : 4,
    "sixOpened" : 5,
    "sevenOpened" : 6,
    "eigthOpened" : 7,
    "nineOpened" : 8
}

closeLabels = {
    "oneClosed" : 0,
    "twoClosed" : 1,
    "threeClosed" : 2,
    "fourClosed" : 3,
    "fiveClosed" : 4,
    "sixClosed" : 5,
    "sevenClosed" : 6,
    "eigthClosed" : 7,
    "nineClosed" : 8
}

diceLabels = {
    "dice1" : 1,
    "dice2" : 2,
    "dice3" : 3,
    "dice4" : 4,
    "dice5" : 5,
    "dice6" : 6
}

class ShutTheBoxGame:
    def __init__(self):
        self.tiles_state_ = [None]*9
        self.dice_ = 0
        self.move_ = []
        self.playerHasToPlay_ = False
        self.endTheGame_ = False
        self.isGameStarted_ = False
        self.diceRolled_ = False

    def waitForDiceToBeRolled(self):
        # Check for the 'q' key to exit
        print("Press space key when rolled the dice")
        if cv2.waitKey(1) & 0xFF == ord(' '):
            self.diceRolled_ = True
        
        
    def startGame(self):
        if None in self.tiles_state_ or False in self.tiles_state_:
            print("To start the game open all the tiles")
            self.isGameStarted_ = False
            time.sleep(0.1)
        else:
            self.isGameStarted_ =  True
            time.sleep(0.1)
    
    def updateGameState(self,tiles,dices):
        if tiles == None or dices == None:
            return
        for state in tiles:
            if state in openLabels.keys():
                self.tiles_state_[openLabels[state]]=True
            elif state in closeLabels.keys():
                self.tiles_state_[closeLabels[state]]=False
        if not self.diceRolled_:
            self.dice_ = 0
            for dice in dices:
                self.dice_ += diceLabels[dice]
        
    def playerPlayed(self):
        for i in self.move_:
            if self.tiles_state_[i-1] != False:
                print(f"waiting for the player to play. Dice value {self.dice_} Close following tiles: {self.move_}")
                time.sleep(0.1)
                return
        self.playerHasToPlay_ = False
        self.diceRolled_ = False

           

    
    def shut_the_box_move(self):
        """
        Determines and suggests a valid move in the 'Shut the Box' game based on the current tile state 
        and the most recent dice roll.

        The method:
        - Checks which tiles are still open.
        - Searches for all combinations of open tiles whose sum equals the current dice roll.
        - If multiple valid moves are found, it chooses the one that closes the most tiles.
        - If no valid moves exist, the game ends.

        This method updates:
        - self.move_: The selected tile combination to close.
        - self.playerHasToPlay_: Flag indicating the player should act.
        - self.endTheGame_: Flag indicating the game is over if no moves are found.
        """

        if self.dice_ == 0:
            print("Waiting to roll the dice")
            time.sleep(0.1)
            return
        
        available_tiles = [i + 1 for i, is_open in enumerate(self.tiles_state_) if is_open]
        valid_moves = []

        def find_combinations(current, start):
            # Calculate the sum of the current combination.
            current_sum = sum(current)
            # If the current sum equals the target dice roll, record this valid move.
            if current_sum == self.dice_:
                valid_moves.append(current.copy())
                return
            # If the sum exceeds the dice roll, no point in continuing.
            if current_sum > self.dice_:
                return
            # Try adding each available tile (maintaining increasing order to avoid duplicates)
            for i in range(start, len(available_tiles)):
                current.append(available_tiles[i])
                find_combinations(current, i + 1)
                current.pop()

        # Initiate the recursive search for valid moves.
        find_combinations([], 0)

        # Return None if no valid move exists.
        if not valid_moves:
            print("No valid moves available, your score is : ", sum(available_tiles))
            self.endTheGame_ = True
            return

        # Heuristic: choose the combination that uses the maximum number of tiles.
        self.move_ = max(valid_moves, key=lambda move: len(move))
        print("Close following tiles : ",self.move_)
        self.playerHasToPlay_=True


class CameraPredictor:
    def __init__(self, model_path, camera_id=0):
        # Load your trained YOLOv8 model
        self.model = YOLO(model_path)
        
        # Open the camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Unable to open camera.")
        
        # Latest prediction and frame will be stored here
        self.latest_prediction = None
        self.latest_frame = None
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.running = True
        
        # Start the background thread for capturing and predicting
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Continuously capture frames, perform inference, and store results."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue  # Skip if we couldn't read a frame
            
            # Run YOLOv8 inference on the current frame.
            results = self.model(frame, verbose=False)
            
            # Store both the latest frame and prediction in a thread-safe manner.
            with self.lock:
                self.latest_prediction = results
                self.latest_frame = frame
            
            # Small sleep to reduce CPU load.
            time.sleep(0.01)

    def get_latest_prediction(self):
        """Return the most recent prediction result."""
        with self.lock:
            return self.latest_prediction

    def get_latest_frame(self):
        """Return the most recent captured frame."""
        with self.lock:
            return self.latest_frame

    def stop(self):
        """Stop the capture thread and release the camera."""
        self.running = False
        self.thread.join()
        self.cap.release()


    def process_prediction(self, prediction, names):
        """
        Processes predictions and extracts detected dice and tile labels.

        For each prediction frame, it checks if there are detections.
        For each detection, it distinguishes between dice and tiles using the provided 
        'names' mapping (class IDs to label names). The function returns a tuple:
          (list of dice labels, list of tile labels)

        It expects exactly 2 dice and 9 tiles to be detected. If a frame does not meet these 
        counts, it continues to the next frame.

        Parameters:
            prediction: An iterable of prediction objects, each having a 'boxes.data'
                        attribute (assumed to be a NumPy array with shape [n, 6]).
            names (dict): A dictionary mapping class IDs to their corresponding label strings.

        Returns:
            tuple: (dice_detected, tiles_detected) if a frame with exactly 2 dice and 9 tiles is found;
                   otherwise, (None, None).
        """

        if not prediction:
            print("No prediction available yet.")
            return None, None

        for result in prediction:
            # Check if any detections exist in this frame.
            if result.boxes.data.shape[0] == 0:
                print("No detections in this frame.")
                continue
            
            # Each detection is assumed to be:
            # [x1, y1, x2, y2, confidence, class_id]
            pred_boxes = result.boxes.data.tolist()
            tiles_detected = []
            dice_detected = []

            for bbox in pred_boxes:
                class_id = int(bbox[-1])
                # Use the names dictionary for a more descriptive label.
                label = names.get(class_id, str(class_id))
                # Convert the label to lower case for a robust 'dice' check.
                if 'dice' in label.lower():
                    dice_detected.append(label)
                else:
                    tiles_detected.append(label)

            # If the detection counts meet the expected numbers, return them.
            if len(dice_detected) == 2 and len(tiles_detected) == 9:
                return dice_detected, tiles_detected

        # If no frame meets the expected detection counts, notify and return (None, None).
        print("No frame with the required detections found.")
        return None, None



def draw_detections(frame, prediction, names):
    """
    Draw bounding boxes and labels on the frame based on the prediction.
    """
    if prediction is not None:
        for result in prediction:
            # If no detections, skip drawing.
            if result.boxes.data.shape[0] == 0:
                continue

            # Iterate over each detection.
            for box in result.boxes.data.tolist():
                # Unpack bounding box coordinates, confidence, and class_id
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = names.get(int(cls), str(cls))
                if label in openLabels.keys():
                    pose = [x1, y1 - 10]
                    text = f'open({openLabels[label]+1})'
                    colour = (0,255,0)
                elif label in closeLabels.keys():
                    pose = [x1, y1 + 140]
                    text = f'close({closeLabels[label]+1})'
                    colour = (0,0,255)
                else:
                    pose = [x1, y1 - 10]
                    text = f'{diceLabels[label]}'
                    colour = (255,0,0)
                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                # Put the label and confidence on the frame
                cv2.putText(frame, text, (pose[0],pose[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                
    return frame


if __name__ == '__main__':
    # Initialize the predictor with your trained model (adjust model path and camera id as needed)
    predictor = CameraPredictor('model/best.pt', camera_id=2)
    game = ShutTheBoxGame()
    try:    
        while not game.endTheGame_:
            # Get the latest prediction and frame
            prediction = predictor.get_latest_prediction()
            dice_pred,tiles_pred = predictor.process_prediction(prediction,predictor.model.names)
            frame = predictor.get_latest_frame()
            # print(dice_pred)
            # print(tiles_pred)
            game.updateGameState(tiles=tiles_pred,dices=dice_pred)
            if not game.isGameStarted_:
                print("We are going to start the game")
                game.startGame()
            elif not game.diceRolled_:
                game.waitForDiceToBeRolled()
            elif game.playerHasToPlay_:
                game.playerPlayed()
            else:
                game.shut_the_box_move()
            

            # If a frame exists, draw detections and show it in a window.
            if frame is not None:
                annotated_frame = frame.copy()
                annotated_frame = draw_detections(annotated_frame, prediction, predictor.model.names)
                cv2.imshow("Camera Output", annotated_frame)
            
            # Check for the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # A tiny sleep for responsiveness
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Stopping camera prediction.")
    
    finally:
        predictor.stop()
        cv2.destroyAllWindows()
