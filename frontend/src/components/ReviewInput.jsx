import { useState } from "react";
import api from "../services/api";

function ReviewInput({ setResult }) {

    const [review, setReview] = useState("");

    const [loading, setLoading] = useState(false);

    async function analyzeReview() {

        if(review.trim()===""){
            alert("Please enter a review.");
            return;
        }

        setLoading(true);

        try{

            const response = await api.post("/predict",{
                text:review
            });

            setResult(response.data);

        }
        catch(error){

            console.log(error);

            alert("Backend Error");

        }

        setLoading(false);

    }

    return(

        <div>

            <textarea

                rows="8"

                placeholder="Write your movie review..."

                value={review}

                onChange={(e)=>setReview(e.target.value)}

            />

            <br/>

            <button onClick={analyzeReview}>

                {
                    loading ? "Analyzing..." : "Analyze"
                }

            </button>

        </div>

    );

}

export default ReviewInput;