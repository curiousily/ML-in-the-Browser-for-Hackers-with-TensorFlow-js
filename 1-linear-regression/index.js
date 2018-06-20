function fibonacci(num){
    var a = 1, b = 0, temp;
    var seq = []

    while (num > 0){
        temp = a;
        a = a + b;
        b = temp;
        seq.push(b)
        num--;
    }

    return seq;
}

const seq = fibonacci(100)

console.log(seq)

const xs = tf.tensor1d(seq.slice(0, seq.length - 1))
const ys = tf.tensor1d(seq.slice(1))

const xmin = xs.min();
const xmax = xs.max();
const xrange = xmax.sub(xmin);

function norm(x) {
    return x.sub(xmin).div(xrange);
}

xs.print()
ys.print()

//console.log(xs)
//console.log(ys)

//xs.print()
//ys.print()


xsNorm = norm(xs)
ysNorm = norm(ys)

//xsNorm.print()
//ysNorm.print()


////const w = tf.variable(tf.truncatedNormal([1], mean=0.0, stddev=1.0))
////const b = tf.variable(tf.zeros([1]))

const w = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))

//w.print()
//b.print()

function predict(x) {
  return tf.tidy(() => {
    return x.mul(w).add(b)
  });
}

function loss(predictions, labels) {
  return predictions.sub(labels).square().mean();
}


const numIterations = 10000;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

for (let iter = 0; iter < numIterations; iter++) {
  optimizer.minimize(() => {
      const predsYs = predict(xsNorm);
      const e = loss(predsYs, ysNorm);
      //e.print()
      return e
  });
}

w.print()
b.print()

// Predict on new value

console.log(seq[seq.length - 1])

xTest = tf.tensor1d([2, 573147844013817200000])
predict(xTest).print()
