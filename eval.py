import os
correct = 0
results_dir = 'runs/detect/exp7/labels/' # Edit here for the path where test results are stored.
for fname in os.listdir(results_dir):
  with open(os.path.join(results_dir, fname)) as detect_file:
    detected = list() # [line.split()[0] for line in detect_file.readlines()]
    with open(os.path.join('../test/labels/', fname)) as exp_file:
      expected = [line.split()[0] for line in exp_file.readlines()]
      for line in detect_file.readlines():
        if float(line.split()[-1].strip()) > 0.6:
          detected.append(line.split()[0])
      if set(expected) != set(detected): # We are looking for an exact match in labels/classes.
        pass
        #print(fname, expected, detected)
      else: correct += 1
print(correct)
