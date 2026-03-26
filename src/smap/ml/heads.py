from __future__ import annotations
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from pydantic import BaseModel, Field

def _sigmoid(value):
    clipped = max(min(value, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-clipped))

def _softmax(values):
    if not values:
        return []
    pivot = max(values)
    exponentials = [math.exp(value - pivot) for value in values]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]

def _materialize_features(feature_names, features):
    return [float(features.get(name, 0.0)) for name in feature_names]

class BinaryLinearHeadDefinition(BaseModel):
    name: str
    version: str
    feature_names: list[str]
    bias: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)

class MulticlassLinearHeadDefinition(BaseModel):
    name: str
    version: str
    feature_names: list[str]
    labels: list[str]
    biases: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, dict[str, float]] = Field(default_factory=dict)

class BinaryLinearHead:

    def __init__(self, definition):
        self.definition = definition

    def score(self, features):
        raw = self.definition.bias
        for name in self.definition.feature_names:
            raw += self.definition.weights.get(name, 0.0) * float(features.get(name, 0.0))
        return round(_sigmoid(raw), 6)

class MulticlassLinearHead:

    def __init__(self, definition):
        self.definition = definition

    def score_distribution(self, features):
        raws: list[float] = []
        for label in self.definition.labels:
            raw = self.definition.biases.get(label, 0.0)
            weights = self.definition.weights.get(label, {})
            for name in self.definition.feature_names:
                raw += weights.get(name, 0.0) * float(features.get(name, 0.0))
            raws.append(raw)
        probabilities = _softmax(raws)
        return {label: round(probability, 6) for label, probability in zip(self.definition.labels, probabilities, strict=True)}

    def predict(self, features):
        distribution = self.score_distribution(features)
        label = max(distribution.items(), key=lambda item: item[1])[0]
        return (label, distribution)

def load_binary_linear_head(path):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return BinaryLinearHead(BinaryLinearHeadDefinition.model_validate(payload))

def load_multiclass_linear_head(path):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return MulticlassLinearHead(MulticlassLinearHeadDefinition.model_validate(payload))

def save_head_definition(path, definition):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(definition.model_dump_json(indent=2), encoding='utf-8')

def train_binary_logistic(*, name, version, feature_names, samples, epochs=600, learning_rate=0.2, l2=0.0005):
    feature_names = list(feature_names)
    materialized = [(_materialize_features(feature_names, features), int(label)) for features, label in samples]
    weights = [0.0 for _ in feature_names]
    bias = 0.0
    if not materialized:
        return BinaryLinearHeadDefinition(name=name, version=version, feature_names=feature_names)
    sample_count = float(len(materialized))
    for _ in range(max(epochs, 1)):
        gradients = [0.0 for _ in feature_names]
        bias_gradient = 0.0
        for vector, label in materialized:
            raw = bias + sum((weight * value for weight, value in zip(weights, vector, strict=True)))
            probability = _sigmoid(raw)
            error = probability - float(label)
            for index, value in enumerate(vector):
                gradients[index] += error * value + l2 * weights[index]
            bias_gradient += error
        for index in range(len(weights)):
            weights[index] -= learning_rate * (gradients[index] / sample_count)
        bias -= learning_rate * (bias_gradient / sample_count)
    return BinaryLinearHeadDefinition(name=name, version=version, feature_names=feature_names, bias=round(bias, 8), weights={name: round(weight, 8) for name, weight in zip(feature_names, weights, strict=True)})

def train_multiclass_logistic(*, name, version, feature_names, labels, samples, epochs=700, learning_rate=0.15, l2=0.0005):
    feature_names = list(feature_names)
    labels = list(labels)
    label_index = {label: index for index, label in enumerate(labels)}
    materialized = [(_materialize_features(feature_names, features), label_index[label]) for features, label in samples if label in label_index]
    weights = [[0.0 for _ in feature_names] for _ in labels]
    biases = [0.0 for _ in labels]
    if not materialized:
        return MulticlassLinearHeadDefinition(name=name, version=version, feature_names=feature_names, labels=labels)
    sample_count = float(len(materialized))
    for _ in range(max(epochs, 1)):
        gradient_weights = [[0.0 for _ in feature_names] for _ in labels]
        gradient_biases = [0.0 for _ in labels]
        for vector, target_index in materialized:
            raws = [biases[label_position] + sum((weight * value for weight, value in zip(weights[label_position], vector, strict=True))) for label_position in range(len(labels))]
            probabilities = _softmax(raws)
            for label_position in range(len(labels)):
                error = probabilities[label_position] - (1.0 if label_position == target_index else 0.0)
                gradient_biases[label_position] += error
                for feature_index, value in enumerate(vector):
                    gradient_weights[label_position][feature_index] += error * value + l2 * weights[label_position][feature_index]
        for label_position in range(len(labels)):
            biases[label_position] -= learning_rate * (gradient_biases[label_position] / sample_count)
            for feature_index in range(len(feature_names)):
                weights[label_position][feature_index] -= learning_rate * (gradient_weights[label_position][feature_index] / sample_count)
    return MulticlassLinearHeadDefinition(name=name, version=version, feature_names=feature_names, labels=labels, biases={label: round(bias, 8) for label, bias in zip(labels, biases, strict=True)}, weights={label: {feature_name: round(weight, 8) for feature_name, weight in zip(feature_names, label_weights, strict=True)} for label, label_weights in zip(labels, weights, strict=True)})
