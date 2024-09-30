import random
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Analytics prompt template
analytics_prompt = """
You are an AI assistant specializing in healthcare analytics for insurance companies. Based on the disease {disease}, provide the following information:

1. Prevalence: Estimate the prevalence of the disease in the general population.
2. Age Distribution: Describe how the disease affects different age groups.
3. Gender Distribution: Explain any gender-specific trends for this disease.
4. Average Treatment Cost: Estimate the average cost of treatment for this disease.
5. Hospitalization Rate: Provide an estimate of how often this disease requires hospitalization.
6. Comorbidities: List common comorbidities associated with this disease.
7. Prevention Strategies: Suggest effective prevention strategies for this disease.
8. Long-term Prognosis: Describe the long-term outlook for patients with this disease.

For each point, provide specific numbers or percentages where applicable. These will be used to generate visualizations.
"""

# Define prompt template
ANALYTICS_PROMPT = PromptTemplate(template=analytics_prompt, input_variables=["disease"])

# Placeholder for LLM response parsing
def parse_llm_response(response):
    # For this example, we'll use mock data. In a real scenario, you'd parse the response from LLM here.
    age_data = {
        'type': 'bar',
        'labels': ['0-18', '19-40', '41-60', '61+'],
        'values': [random.randint(5, 20), random.randint(20, 40), random.randint(30, 50), random.randint(10, 30)]
    }
    
    gender_data = {
        'type': 'pie',
        'labels': ['Male', 'Female'],
        'values': [random.randint(40, 60), random.randint(40, 60)]
    }
    
    cost_data = {
        'type': 'pie',
        'labels': ['Medication', 'Hospital Stay', 'Outpatient Care', 'Tests'],
        'values': [random.randint(2000, 7000), random.randint(8000, 20000), random.randint(1000, 4000), random.randint(500, 2000)]
    }
    
    comorbidities_data = {
        'type': 'bar',
        'labels': ['Hypertension', 'Diabetes', 'Obesity', 'Heart Disease'],
        'values': [random.randint(20, 60), random.randint(10, 40), random.randint(15, 35), random.randint(10, 25)]
    }
    
    return {
        'age_distribution': age_data,
        'gender_distribution': gender_data,
        'cost_breakdown': cost_data,
        'comorbidities': comorbidities_data
    }

# Main function to handle analytics
def process_analytics(disease_name, get_llama3_llm):
    try:
        # Run LLM chain for disease analytics
        llm = get_llama3_llm()
        chain = LLMChain(llm=llm, prompt=ANALYTICS_PROMPT)
        response = chain.run(disease=disease_name)
        
        # Parse LLM response for visualizations (mock data for now)
        chart_data = parse_llm_response(response)
        
        # Generate a summary report
        summary = f"""
        Based on the analysis of {disease_name}, here are key points for insurance consideration:
        
        1. **Prevalence**: {random.randint(1, 10)}% of the population is affected by this disease.
        2. **Average Treatment Cost**: ${random.randint(5000, 50000)} per year.
        3. **Hospitalization Rate**: {random.randint(10, 50)}% of patients require hospitalization.
        4. **Long-term Prognosis**: {random.choice(['Generally good with proper management', 
           'Variable depending on individual factors', 
           'May require ongoing care and monitoring'])}
        
        **Recommendations**:
        1. Implement targeted prevention programs to reduce incidence and severity.
        2. Adjust premiums based on risk factors and prevention adherence.
        3. Develop partnerships with healthcare providers for cost-effective treatments.
        4. Invest in patient education to improve self-management and reduce complications.
        """
        
        return {
            "analysis": response,
            "summary": summary,
            **chart_data
        }
    except Exception as e:
        return {"error": str(e)}