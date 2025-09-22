/*
 * Copyright 2023 Aryn
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.opensearch.ml.engine.algorithms.text_similarity;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.opensearch.ml.common.output.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.algorithms.SentenceTransformerTranslator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.TranslatorContext;

public class TextSimilarityTranslator extends SentenceTransformerTranslator {
    public final String SIMILARITY_NAME = "similarity";

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        super.prepare(ctx);
    }

    @Override
    public ai.djl.translate.Batchifier getBatchifier() {
        return ai.djl.translate.Batchifier.STACK;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        String sentence = input.getAsString(0);
        String context = input.getAsString(1);
        
        NDManager manager = ctx.getNDManager();
        NDList ndList = new NDList();
        Encoding encodings = tokenizer.encode(sentence, context);
        long[] indices = encodings.getIds();
        long[] attentionMask = encodings.getAttentionMask();
        long[] tokenTypes = encodings.getTypeIds();

        NDArray indicesArray = manager.create(indices);
        indicesArray.setName("input_ids");

        NDArray attentionMaskArray = manager.create(attentionMask);
        attentionMaskArray.setName("attention_mask");

        NDArray tokenTypeArray = manager.create(tokenTypes);
        tokenTypeArray.setName("token_type_ids");

        ndList.add(indicesArray);
        ndList.add(attentionMaskArray);
        ndList.add(tokenTypeArray);
        
        return ndList;
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        Output output = new Output(200, "OK");

        List<ModelTensor> outputs = new ArrayList<>();
        Iterator<NDArray> iterator = list.iterator();
        while (iterator.hasNext()) {
            NDArray ndArray = iterator.next();
            
            String name = SIMILARITY_NAME;
            Number[] data = ndArray.toArray();
            long[] shape = ndArray.getShape().getShape();
            DataType dataType = ndArray.getDataType();
            MLResultDataType mlResultDataType = MLResultDataType.valueOf(dataType.name());
            ByteBuffer buffer = ndArray.toByteBuffer();
            
            ModelTensor tensor = ModelTensor
                .builder()
                .name(name)
                .data(data)
                .shape(shape)
                .dataType(mlResultDataType)
                .byteBuffer(buffer)
                .build();
            outputs.add(tensor);
        }

        ModelTensors modelTensorOutput = new ModelTensors(outputs);
        output.add(modelTensorOutput.toBytes());
        
        return output;
    }

    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<Input> inputs) {
        NDManager manager = ctx.getNDManager();
        
        // Extract text pairs
        List<String> sentences = new ArrayList<>();
        List<String> contexts = new ArrayList<>();
        
        for (int i = 0; i < inputs.size(); i++) {
            Input input = inputs.get(i);
            String sentence = input.getAsString(0);
            String context = input.getAsString(1);
            sentences.add(sentence);
            contexts.add(context);
        }
        
        // Encode all pairs and find max length
        List<Encoding> encodings = new ArrayList<>();
        int maxLength = 0;
        
        for (int i = 0; i < sentences.size(); i++) {
            Encoding encoding = tokenizer.encode(sentences.get(i), contexts.get(i));
            encodings.add(encoding);
            int currentLength = encoding.getIds().length;
            maxLength = Math.max(maxLength, currentLength);
        }
        
        // Create padded tensors
        List<long[]> allIndices = new ArrayList<>();
        List<long[]> allAttentionMasks = new ArrayList<>();
        List<long[]> allTokenTypes = new ArrayList<>();
        
        for (int i = 0; i < encodings.size(); i++) {
            Encoding encoding = encodings.get(i);
            long[] indices = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();
            long[] tokenTypes = encoding.getTypeIds();
            
            // Pad to maxLength
            long[] paddedIndices = new long[maxLength];
            long[] paddedAttentionMask = new long[maxLength];
            long[] paddedTokenTypes = new long[maxLength];
            
            System.arraycopy(indices, 0, paddedIndices, 0, indices.length);
            System.arraycopy(attentionMask, 0, paddedAttentionMask, 0, attentionMask.length);
            System.arraycopy(tokenTypes, 0, paddedTokenTypes, 0, tokenTypes.length);
            
            allIndices.add(paddedIndices);
            allAttentionMasks.add(paddedAttentionMask);
            allTokenTypes.add(paddedTokenTypes);
        }
        
        // Create batch tensors
        NDArray indicesArray = manager.create(allIndices.toArray(new long[0][]));
        indicesArray.setName("input_ids");
        
        NDArray attentionMaskArray = manager.create(allAttentionMasks.toArray(new long[0][]));
        attentionMaskArray.setName("attention_mask");
        
        NDArray tokenTypeArray = manager.create(allTokenTypes.toArray(new long[0][]));
        tokenTypeArray.setName("token_type_ids");
        
        NDList ndList = new NDList();
        ndList.add(indicesArray);
        ndList.add(attentionMaskArray);
        ndList.add(tokenTypeArray);
        
        return ndList;
    }

    @Override
    public List<Output> batchProcessOutput(TranslatorContext ctx, NDList list) {
        List<Output> outputs = new ArrayList<>();
        
        // Process all NDArrays in the list (similar to processOutput)
        if (list.isEmpty()) {
            return outputs;
        }
        
        // Get the first NDArray to determine batch size
        NDArray firstArray = list.get(0);
        long[] shape = firstArray.getShape().getShape();
        int batchSize = (int) shape[0];
        
        // Create outputs for each item in the batch
        for (int i = 0; i < batchSize; i++) {
            Output output = new Output(200, "OK");
            List<ModelTensor> tensorList = new ArrayList<>();
            
            // Process all NDArrays in the list for this batch item
            for (int j = 0; j < list.size(); j++) {
                NDArray ndArray = list.get(j);
                NDArray singleResult = ndArray.get(i);
                
                String name = SIMILARITY_NAME;
                Number[] data = singleResult.toArray();
                long[] resultShape = singleResult.getShape().getShape();
                DataType dataType = singleResult.getDataType();
                MLResultDataType mlResultDataType = MLResultDataType.valueOf(dataType.name());
                ByteBuffer buffer = singleResult.toByteBuffer();
                
                ModelTensor tensor = ModelTensor
                    .builder()
                    .name(name)
                    .data(data)
                    .shape(resultShape)
                    .dataType(mlResultDataType)
                    .byteBuffer(buffer)
                    .build();
                tensorList.add(tensor);
            }
            
            ModelTensors modelTensorOutput = new ModelTensors(tensorList);
            output.add(modelTensorOutput.toBytes());
            outputs.add(output);
        }
        
        return outputs;
    }

}
