from IPython.display import Image, display
from graphviz import Digraph

def display_nn_graph(model, filename):
    """
    Display a detailed visualization of a neural network architecture
    """
    
    def get_layer_info(layer):
        """Extract relevant information from different layer types"""
        info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'units': None,
            'activation': 'N/A',
            'color': 'lightgreen'
        }
        
        # Dense layers
        if hasattr(layer, 'units'):
            info['units'] = layer.units
            
        # Convolutional layers (Conv1D, Conv2D, Conv3D)
        elif hasattr(layer, 'filters'):
            info['units'] = layer.filters
            
        # Recurrent layers (LSTM, GRU, SimpleRNN)
        elif 'LSTM' in info['type'] or 'GRU' in info['type'] or 'RNN' in info['type']:
            if hasattr(layer, 'units'):
                info['units'] = layer.units
            else:
                info['units'] = layer.output_shape[-1] if hasattr(layer, 'output_shape') else 1
                
        # Pooling layers
        elif 'Pool' in info['type']:
            info['units'] = None  # Pooling doesn't have units
            info['color'] = 'lightyellow'
            
        # Dropout, BatchNormalization, etc.
        elif info['type'] in ['Dropout', 'BatchNormalization', 'LayerNormalization']:
            info['units'] = None
            info['color'] = 'lightgray'
            
        # Flatten, Reshape
        elif info['type'] in ['Flatten', 'Reshape']:
            info['units'] = None
            info['color'] = 'lightcyan'
            
        # Embedding layers
        elif 'Embedding' in info['type']:
            info['units'] = layer.output_dim if hasattr(layer, 'output_dim') else None
            info['color'] = 'lightpink'
            
        # Fallback to output shape
        if info['units'] is None and hasattr(layer, 'output_shape'):
            shape = layer.output_shape
            if isinstance(shape, tuple) and len(shape) > 1:
                info['units'] = shape[-1]
            elif isinstance(shape, list) and len(shape) > 0:
                info['units'] = shape[0][-1] if isinstance(shape[0], tuple) else shape[0]
        
        # Default units if still None
        if info['units'] is None:
            info['units'] = 1
            
        # Get activation function
        if hasattr(layer, 'activation'):
            if hasattr(layer.activation, '__name__'):
                info['activation'] = layer.activation.__name__
            else:
                info['activation'] = str(layer.activation).split()[0].replace('<', '')
                
        return info
    
    def create_nn_graph(model, filename="detailed_nn"):
        """Create a detailed neural network visualization"""
        title = model.name if hasattr(model, 'name') else "Neural Network"
        print(f"Creating visualization for model: {title}")
        
        dot = Digraph(comment=title, format='png')
        dot.attr(rankdir='LR', size='12,8', splines='true')
        dot.attr('graph', fontsize='16', label=title, labelloc='top')
        dot.attr('node', shape='circle', style='filled', fontsize='10', 
                width='0.3', height='0.3')
        
        max_neurons_display = 8
        connection_limit = 20
        
        # Input layer
        print(f"Processing input layer...")
        first_layer_info = get_layer_info(model.layers[0])
        input_dim = first_layer_info['units']
        input_display = min(input_dim, max_neurons_display)
        
        input_nodes = []
        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='Input Layer', style='dashed')
            for i in range(input_display):
                node_id = f"input_{i}"
                c.node(node_id, '', fillcolor='lightblue')
                input_nodes.append(node_id)
            if input_dim > max_neurons_display:
                c.node('input_more', '...', shape='plaintext')
        
        prev_nodes = input_nodes
        
        # Process all layers
        for layer_idx, layer in enumerate(model.layers):
            layer_info = get_layer_info(layer)
            layer_name = f"layer_{layer_idx}"
            
            print(f"Processing layer {layer_idx}: {layer_info['type']}")
            
            # Skip layers without neurons (like some transformations)
            if layer_info['units'] == 0:
                continue
                
            units = layer_info['units']
            display_units = min(units, max_neurons_display)
            
            current_nodes = []
            with dot.subgraph(name=f'cluster_{layer_name}') as c:
                # Create label
                label_parts = [layer_info['type']]
                if units > 0:
                    label_parts.append(f"({units} units)")
                if layer_info['activation'] != 'N/A':
                    label_parts.append(layer_info['activation'])
                    
                c.attr(label='\\n'.join(label_parts), style='dashed')
                
                # Determine color (output layer gets special color)
                color = 'orange' if layer_idx == len(model.layers) - 1 else layer_info['color']
                
                for i in range(display_units):
                    node_id = f"{layer_name}_{i}"
                    c.node(node_id, '', fillcolor=color)
                    current_nodes.append(node_id)
                
                if units > max_neurons_display:
                    c.node(f'{layer_name}_more', '...', shape='plaintext')
            
            # Connect to previous layer
            connections_made = 0
            for prev_node in prev_nodes:
                for curr_node in current_nodes:
                    if connections_made < connection_limit:
                        dot.edge(prev_node, curr_node, style='solid', arrowsize='0.5')
                        connections_made += 1
                    else:
                        break
                if connections_made >= connection_limit:
                    break
            
            prev_nodes = current_nodes
        
        # Save and render
        dot.render(filename, cleanup=True)
        print(f"Visualization saved as '{filename}.png'")
        return dot
    
    create_nn_graph(model, filename)
    display(Image(f"{filename}.png"))