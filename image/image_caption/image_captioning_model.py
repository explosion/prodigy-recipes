"""
Simple image captioning model
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
"""
import torch
import pickle
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import base64
import io


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VocabUnpickler(pickle.Unpickler):
    # https://stackoverflow.com/a/50472787/6400719
    def find_class(self, module, name):
        if module == "__main__":
            module = "image_captioning_model"
        return super().find_class(module, name)


def b64_uri_to_bytes(data_uri):
    """Convert a base64-encoded data URI to bytes."""
    data = data_uri.split("base64,", 1)[1]
    return base64.decodestring(bytes(data, "ascii"))


def load_model(
    vocab_path="vocab.pkl",
    encoder_path="encoder-5-3000.pkl",
    decoder_path="decoder-5-3000.pkl",
    embed_size=256,
    hidden_size=512,
    num_layers=1,
):
    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Load vocabulary wrapper
    with open(vocab_path, "rb") as f:
        unpickler = VocabUnpickler(f)
        vocab = unpickler.load()
    # Build models, eval mode (batchnorm uses moving mean/variance)
    encoder = EncoderCNN(embed_size).eval()
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder, vocab, transform


def generate_caption(image, encoder, decoder, vocab, transform=None):
    # Instead of loading from a base64 string and bytes, you could also
    # call Image.open with a file path instead
    image_bytes = b64_uri_to_bytes(image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    image_tensor = image.to(device)
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    # Remove <start> and <end> markers
    caption = sampled_caption[1:-1]
    # Remove fullstop
    if caption[-1] == ".":
        caption = caption[:-1]
    return " ".join(caption)


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(
        self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20
    ):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            # hiddens:
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # (batch_size, max_seq_length)
        return sampled_ids


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
