import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load trained model and tokenizer
model_path = "./t5_spell_corrector"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Function to predict corrections
def correct_text(text):
    input_text = f"fix: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(inputs.input_ids, max_length=128)
    
    # Decode the corrected text
    corrected_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return corrected_text

# Example usage
# Given source text (with typos)
src_text = """
// The follwing form of the range check is equivalent but assumes that
"boilerplate to get working."
* for both bitcoind and bitcoin-core, to make it harder for attackers to
Websocket transport implements a HTTP(S) compliable, surveillance proof transport method with plausible deniability.
context "when #bactrace is redefined" do
921817fefea5        node-0/bridge2         brige
"assert(code.match(/console\\.log\\(outputTwo\\)/g), 'Use <code>console.log()</code> to print the <code>outputTwice</code> variable.  In your Browser Console this should print out the value of the variable two times.');"
Th documentation is available at [link]().
* Who terminted my EC2 instance?
After NumPy is installed, install scipy since some of the plots in the random
- bind-token-to
"%s.%s" % (scenario.__name__, method)
Fixed minimize-to-dock behavior of Bitcon-Qt on the Mac.
and enforces access controls on which parts of that code
2. `1.5.0` - for Angular JS 1.5 and above which supports native `.componet(name, options)` API
/** Sets up the the trigger event listeners if ripples are enabled. */
*NN_TTL*::
if is-interative; then
.CreatedSince - Elapsed time since the image was created.
"""

# Use the trained model to correct the text
corrected_text = correct_text(src_text)

# Print results
print("🔹 Input (with errors):")
print(src_text)

print("\n✅ Corrected Output:")
print(corrected_text)

