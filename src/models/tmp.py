from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('batch_regression_model_best.h5')

plt.plot(model.history['loss'], label='Train Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Regression Model Accuracy')
plt.legend()
plt.savefig('regression.png')