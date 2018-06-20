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

const fibs = fibonacci(100)

//console.log(fibs)

const xs = tf.tensor1d(fibs.slice(0, fibs.length - 1))
const ys = tf.tensor1d(fibs.slice(1))

const xmin = xs.min();
const xmax = xs.max();
const xrange = xmax.sub(xmin);

function norm(x) {
    return x.sub(xmin).div(xrange);
}

//xs.print()
//ys.print()

//console.log(xs)
//console.log(ys)

//xs.print()
//ys.print()


xsNorm = norm(xs)
ysNorm = norm(ys)

//console.log(xsNorm.dataSync())
//console.log(ysNorm.dataSync())


const a = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))

//w.print()
//b.print()

function predict(x) {
    return tf.tidy(() => {
        return a.mul(x).add(b)
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

a.print()
b.print()

// Predict on new value

console.log(fibs[fibs.length - 1])

xTest = tf.tensor1d([2, 354224848179262000000])
predict(xTest).print()
