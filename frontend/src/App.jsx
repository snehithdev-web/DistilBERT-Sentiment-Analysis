import { useState } from "react";

import ReviewInput from "./components/ReviewInput";
import ResultCard from "./components/ResultCard";

function App() {

    const [result, setResult] = useState(null);

    return (
        <div className="app">

            <h1>🎬 DistilBERT Sentiment Analysis</h1>

            <p>
                Analyze movie reviews using a fine-tuned DistilBERT model.
            </p>

            <ReviewInput setResult={setResult} />

            {
                result && <ResultCard result={result}/>
            }

        </div>
    );
}

export default App;