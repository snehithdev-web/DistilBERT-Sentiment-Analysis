function ResultCard({ result }) {

    return(

        <div>

            <h2>Prediction</h2>

            <h3>{result.sentiment}</h3>

            <h4>

                Confidence :

                {result.confidence.toFixed(2)}%

            </h4>

        </div>

    );

}

export default ResultCard;