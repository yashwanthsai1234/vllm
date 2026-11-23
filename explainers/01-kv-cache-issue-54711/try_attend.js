// Your turn: write attention yourself. It is about 8 lines.
//
// Inputs (the numbers from the video):
//   q = the query of "sat"                  e.g. [2, 0]
//   K = the keys, one row per token         e.g. [[1, 0], [0, 1], [1, 1]]
//   V = the values, one row per token       e.g. [[1, 0], [0, 1], [2, 2]]
//
// Return the output vector. For the inputs above it is about [1.3374, 1.0].
// (The video shows 1.338 because it adds numbers that were already rounded.)
//
// The three steps:
//   Step 1  score_i  = (q · k_i) / sqrt(d)        d = q.length
//   Step 2  share_i  = e^(score_i) / (sum of all e^(score_j))
//   Step 3  output   = sum of share_i × v_i       (do it for each number in the vector)
//
// Helpers you may want: Math.sqrt, Math.exp, Array.map, Array.reduce.
// Save the file, reload index.html, and move the sliders: the badge turns green when
// your answer matches for every query you try.

function attend(q, K, V) {
  // TODO: replace this line with your code.
  return null;
}
