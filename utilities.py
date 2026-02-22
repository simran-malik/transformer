
import matplotlib.pyplot as plt
import torch
import os

class Utilities:
    def __init__(self, tokenizer, model, directory, plot_name_suffix):
        self.tokenizer = tokenizer
        self.model = model
        self.plot_name_suffix = plot_name_suffix
        self.directory = directory

    def sanity_check(self, sentence, block_size):
        os.makedirs(f"{self.directory}", exist_ok=True)

        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            selected_head = attn_map[0, 0] 
            att_map = selected_head.detach().cpu().numpy()
            total_prob_over_rows = torch.sum(selected_head, dim=1)

            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.detach().cpu().numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")
            
            # Save the plot
            plt.savefig(f"{self.directory}/attention_map_{j + 1}_{self.plot_name_suffix}.png")
            
            # Show the plot
            plt.show()
            


