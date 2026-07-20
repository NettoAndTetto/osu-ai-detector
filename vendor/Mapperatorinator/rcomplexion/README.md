# RComplexion

RComplexion is an attempt to estimate the complexity of rhythm in osu! beatmaps using cross entropy.
A deep model is trained to predict the timing of the next note in a beatmap given the timing of the previous notes.
The theory is that the more complex the rhythm, the harder it is to predict the timing of the next note.

See `datasets/rhythm_complexities.csv` for the estimated complexities of some beatmaps.
The first column is the beatmap ID, and the second column is the estimated complexity.
